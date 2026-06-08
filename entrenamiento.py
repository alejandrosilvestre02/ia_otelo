import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import random
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

try:
	import tensorflow as tf  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - depende del entorno
	tf = None  # type: ignore[assignment]
	_TENSORFLOW_IMPORT_ERROR = exc
else:
	_TENSORFLOW_IMPORT_ERROR = None

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales, nuevo_tablero


random.seed(394867)
np.random.seed(394867)
if tf is not None:
	tf.keras.utils.set_random_seed(394867)


LABEL_LOSS = 0
LABEL_DRAW = 1
LABEL_WIN = 2
DEFAULT_MODEL_PATH = Path("modelo_otelo.keras")


def codifica_tablero(tablero: Sequence[Sequence[int]], jugador: int) -> np.ndarray:
	vector = np.zeros(64, dtype=np.float32)
	indice = 0
	for fila in tablero:
		for celda in fila:
			if celda == jugador:
				vector[indice] = 1.0
			elif celda != 0:
				vector[indice] = -1.0
			indice += 1
	return vector


def resultado_desde_perspectiva(negro: int, blanco: int, jugador: int) -> int:
	if negro == blanco:
		return LABEL_DRAW
	if jugador == 2:
		return LABEL_WIN if negro > blanco else LABEL_LOSS
	return LABEL_WIN if blanco > negro else LABEL_LOSS


def simula_partida_aleatoria() -> Tuple[List[np.ndarray], List[int]]:
	tablero = nuevo_tablero()
	jugador = 2
	historico: List[Tuple[np.ndarray, int]] = []
	pasadas_consecutivas = 0

	while pasadas_consecutivas < 2:
		movimientos = movimientos_legales(tablero, jugador)
		if not movimientos:
			pasadas_consecutivas += 1
			jugador = 3 - jugador
			continue

		pasadas_consecutivas = 0
		historico.append((codifica_tablero(tablero, jugador), jugador))
		from algoritmo_uct import selecciona_movimiento as seleccion_UCT
		movimiento, fichas_volteadas = seleccion_UCT(tablero, jugador, modelo=None, iteraciones=50)
		aplicar_movimiento(tablero, movimiento, jugador, fichas_volteadas)
		jugador = 3 - jugador

	negro, blanco = cuenta_fichas(tablero)
	entradas = [entrada for entrada, _ in historico]
	labels = [resultado_desde_perspectiva(negro, blanco, jugador_historico) for _, jugador_historico in historico]
	return entradas, labels


def genera_dataset(num_partidas: int) -> Tuple[np.ndarray, np.ndarray]:
	entradas: List[np.ndarray] = []
	labels: List[int] = []

	for _ in range(num_partidas):
		x_partida, y_partida = simula_partida_aleatoria()
		entradas.extend(x_partida)
		labels.extend(y_partida)

	if not entradas:
		raise RuntimeError("No se generaron ejemplos de entrenamiento.")

	x = np.stack(entradas).astype(np.float32)
	y = np.asarray(labels, dtype=np.int64)
	return x, y


class RedNeuronalOtelo:
	"""Red neuronal Keras para clasificar el resultado de una posición de Otelo."""

	def __init__(self, dimensiones: Sequence[int] = (64, 128, 64, 3), seed: int = 394867, modelo: Any = None) -> None:
		if tf is None:
			raise ImportError(
				"TensorFlow no está disponible en este entorno. Usa un Python compatible con TensorFlow (por ejemplo, 3.13) e instala el paquete tensorflow."
			) from _TENSORFLOW_IMPORT_ERROR
		self.dimensiones = list(dimensiones)
		self.seed = seed
		self.modelo = modelo if modelo is not None else self.construye_modelo()

	def construye_modelo(self):
		initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)
		modelo = tf.keras.Sequential(
			[
				tf.keras.layers.Input(shape=(self.dimensiones[0],)),
				tf.keras.layers.Dense(self.dimensiones[1], activation="tanh", kernel_initializer=initializer),
				tf.keras.layers.Dense(self.dimensiones[2], activation="tanh", kernel_initializer=initializer),
				tf.keras.layers.Dense(self.dimensiones[3], activation="softmax", kernel_initializer=initializer),
			]
		)
		modelo.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=["accuracy"],
		)
		return modelo

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float32)
		if x.ndim == 1:
			x = x[None, :]
		return self.modelo.predict(x, verbose=0)

	def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 128, learning_rate: float = 0.01) -> None:
		x = np.asarray(x, dtype=np.float32)
		y = np.asarray(y, dtype=np.int64)
		if len(x) != len(y):
			raise ValueError("x e y deben tener la misma longitud")

		self.modelo.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			metrics=["accuracy"],
		)
		self.modelo.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

	def train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
		self.fit(x, y, epochs=1, batch_size=max(len(x), 1), learning_rate=learning_rate)

	@staticmethod
	def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
		indices = np.arange(len(y))
		probabilidades = np.clip(probs[indices, y], 1e-9, 1.0)
		return float(-np.mean(np.log(probabilidades)))

	def save(self, ruta: Path) -> None:
		ruta = Path(ruta)
		if ruta.suffix == "":
			ruta = ruta.with_suffix(".keras")
		self.modelo.save(ruta)

	@classmethod
	def load(cls, ruta: Path) -> "RedNeuronalOtelo":
		if tf is None:
			raise ImportError(
				"TensorFlow no está disponible en este entorno. Usa un Python compatible con TensorFlow (por ejemplo, 3.13) e instala el paquete tensorflow."
			) from _TENSORFLOW_IMPORT_ERROR
		ruta = Path(ruta)
		if not ruta.exists() and ruta.suffix == ".npz":
			ruta = ruta.with_suffix(".keras")
		modelo = tf.keras.models.load_model(ruta)
		return cls(modelo=modelo)


def selecciona_movimiento(modelo: RedNeuronalOtelo, tablero: Sequence[Sequence[int]], jugador: int):
	movimientos = movimientos_legales(tablero, jugador)
	if not movimientos:
		return None

	mejor_movimiento = None
	mejor_puntuacion = float("-inf")
	for movimiento, fichas_volteadas in movimientos.items():
		copia = [fila[:] for fila in tablero]
		aplicar_movimiento(copia, movimiento, jugador, fichas_volteadas)
		jugador_siguiente = 3 - jugador
		probabilidades = modelo.predict_proba(codifica_tablero(copia, jugador_siguiente))[0]
		puntuacion = float(probabilidades[LABEL_LOSS])
		if puntuacion > mejor_puntuacion:
			mejor_puntuacion = puntuacion
			mejor_movimiento = (movimiento, fichas_volteadas)

	return mejor_movimiento


def entrena_modelo(num_partidas: int, epochs: int, batch_size: int, learning_rate: float, ruta_salida: Path) -> RedNeuronalOtelo:
	print(f"Generando datos con {num_partidas} partidas aleatorias...")
	x, y = genera_dataset(num_partidas)
	print(f"Ejemplos generados: {len(x)}")

	modelo = RedNeuronalOtelo()
	modelo.fit(x, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
	modelo.save(ruta_salida)
	print(f"Modelo guardado en {ruta_salida}")
	return modelo


def juega_contra_modelo(ruta_modelo: Path) -> None:
	modelo = RedNeuronalOtelo.load(ruta_modelo)
	tablero = nuevo_tablero()
	jugador = 2

	while True:
		movimientos = movimientos_legales(tablero, jugador)
		jugador_oponente = 3 - jugador
		if not movimientos and not movimientos_legales(tablero, jugador_oponente):
			break

		if not movimientos:
			print(f"El jugador {jugador} pasa.")
			jugador = jugador_oponente
			continue

		seleccion = selecciona_movimiento(modelo, tablero, jugador)
		assert seleccion is not None
		movimiento, fichas_volteadas = seleccion
		color = "Negro" if jugador == 2 else "Blanco"
		print(f"{color} juega {movimiento}")
		aplicar_movimiento(tablero, movimiento, jugador, fichas_volteadas)

		jugador = jugador_oponente

	negro, blanco = cuenta_fichas(tablero)
	print(f"Resultado final - Negro: {negro}  Blanco: {blanco}")


def construye_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Entrena una red neuronal simple para Otelo.")
	subparsers = parser.add_subparsers(dest="comando", required=True)

	train_parser = subparsers.add_parser("train", help="Genera datos y entrena el modelo")
	train_parser.add_argument("--games", type=int, default=300, help="Numero de partidas aleatorias para generar datos")
	train_parser.add_argument("--epochs", type=int, default=15, help="Numero de epocas de entrenamiento")
	train_parser.add_argument("--batch-size", type=int, default=128, help="Tamano del lote")
	train_parser.add_argument("--learning-rate", type=float, default=0.01, help="Tasa de aprendizaje")
	train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Ruta del modelo a guardar")

	play_parser = subparsers.add_parser("play", help="Cargar un modelo y jugar una partida de prueba")
	play_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Ruta del modelo entrenado")

	return parser


def main() -> None:
	parser = construye_parser()
	args = parser.parse_args()

	if args.comando == "train":
		entrena_modelo(args.games, args.epochs, args.batch_size, args.learning_rate, args.output)
	elif args.comando == "play":
		juega_contra_modelo(args.model)


if __name__ == "__main__":
	main()