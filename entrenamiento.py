import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime logs verbosos de TensorFlow (si está instalado)

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales, nuevo_tablero


# Fijar semillas para obtener resultados reproducibles durante pruebas
random.seed(394867)
np.random.seed(394867)


# Etiquetas usadas para el objetivo de la red: pérdida/empate/victoria
LABEL_LOSS = 0
LABEL_DRAW = 1
LABEL_WIN = 2
DEFAULT_MODEL_PATH = Path("modelo_otelo.npz")


def codifica_tablero(tablero: Sequence[Sequence[int]], jugador: int) -> np.ndarray:
	"""Convierte el tablero en un vector de 64 valores desde la perspectiva del jugador actual.

	- 1.0 indica una ficha del jugador
	- -1.0 indica una ficha del oponente
	- 0.0 celda vacía
	Esto sirve como entrada para la red neuronal.
	"""
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
	# Devuelve la etiqueta (LOSS/DRAW/WIN) desde la perspectiva de `jugador`.
	if negro == blanco:
		return LABEL_DRAW
	if jugador == 2:
		return LABEL_WIN if negro > blanco else LABEL_LOSS
	return LABEL_WIN if blanco > negro else LABEL_LOSS


def simula_partida_aleatoria() -> Tuple[List[np.ndarray], List[int]]:
	"""Simula una partida completa con movimientos aleatorios.

	Devuelve una lista de entradas (tableros codificados) y las etiquetas
	correspondientes calculadas al final de la partida.
	"""
	tablero = nuevo_tablero()
	jugador = 2  # empieza el jugador negro (convención interna)
	historico: List[Tuple[np.ndarray, int]] = []
	pasadas_consecutivas = 0

	# El juego termina cuando ambos jugadores pasan consecutivamente
	while pasadas_consecutivas < 2:
		movimientos = movimientos_legales(tablero, jugador)
		if not movimientos:
			# Si no hay movimientos válidos, el jugador pasa
			pasadas_consecutivas += 1
			jugador = 3 - jugador
			continue

		pasadas_consecutivas = 0
		# Guardar la posición y el jugador que la jugó
		historico.append((codifica_tablero(tablero, jugador), jugador))
		# Elegir un movimiento aleatorio entre los legales
		movimiento, fichas_volteadas = list(movimientos.items())[np.random.randint(len(movimientos))]
		aplicar_movimiento(tablero, movimiento, jugador, fichas_volteadas)
		jugador = 3 - jugador

	# Al final de la partida calculamos el resultado y generamos las etiquetas
	negro, blanco = cuenta_fichas(tablero)
	entradas = [entrada for entrada, _ in historico]
	labels = [resultado_desde_perspectiva(negro, blanco, jugador_historico) for _, jugador_historico in historico]
	return entradas, labels


def genera_dataset(num_partidas: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Genera `num_partidas` partidas aleatorias y construye matrices numpy
	`x` (entradas) e `y` (etiquetas) listas para entrenar la red.
	"""
	entradas: List[np.ndarray] = []
	labels: List[int] = []

	for _ in range(num_partidas):
		x_partida, y_partida = simula_partida_aleatoria()
		entradas.extend(x_partida)
		labels.extend(y_partida)

	if not entradas:
		raise RuntimeError("No se generaron ejemplos de entrenamiento.")

	# Apilar en matrices numpy para entrenamiento
	x = np.stack(entradas).astype(np.float32)
	y = np.asarray(labels, dtype=np.int64)
	return x, y


class RedNeuronalOthello:
	"""Implementación mínima de una red feed-forward para clasificar el resultado desde una posición.

	Arquitectura configurable vía `dimensiones`. Las activaciones intermedias usan tanh
	y la salida es softmax sobre 3 clases (LOSS/DRAW/WIN).
	"""
	def __init__(self, dimensiones: Sequence[int] = (64, 128, 64, 3), seed: int = 394867) -> None:
		"""
		Red neuronal densa simple implementada con numpy.
		`dimensiones` define el tamaño de cada capa (entrada,...,salida).
		Se inicializan pesos y sesgos con distribución uniforme (Xavier-like).
		"""
		self.dimensiones = list(dimensiones)
		rag = np.random.default_rng(seed)
		self.pesos = []
		self.sesgos = []
		for entrada, salida in zip(self.dimensiones[:-1], self.dimensiones[1:]):
			# Inicialización basada en el tamaño de las capas para estabilidad
			limite = np.sqrt(6.0 / (entrada + salida))
			self.pesos.append(rag.uniform(-limite, limite, size=(entrada, salida)).astype(np.float32))
			self.sesgos.append(np.zeros(salida, dtype=np.float32))

	@staticmethod
	def _tanh(x: np.ndarray) -> np.ndarray:
		return np.tanh(x)

	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		# Estabilizar antes de exp para evitar overflow
		ajustado = logits - np.max(logits, axis=1, keepdims=True)
		exp = np.exp(ajustado)
		return exp / np.sum(exp, axis=1, keepdims=True)

	def _forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		"""Propagación hacia delante que devuelve activaciones y pre-activaciones.

		- `activaciones` contiene la salida de cada capa (incluida la entrada)
		- `preactivaciones` contiene los valores lineales z = Wx + b por capa
		"""
		activaciones = [x]
		preactivaciones: List[np.ndarray] = []
		capa = x
		for indice, (peso, sesgo) in enumerate(zip(self.pesos, self.sesgos)):
			z = capa @ peso + sesgo
			preactivaciones.append(z)
			if indice == len(self.pesos) - 1:
				# Capa de salida: softmax
				capa = self._softmax(z)
			else:
				# Capas ocultas: tanh
				capa = self._tanh(z)
			activaciones.append(capa)
		return activaciones, preactivaciones

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		# Devuelve las probabilidades (softmax) para las entradas `x`.
		x = np.asarray(x, dtype=np.float32)
		# Aceptar vectores 1D convirtiéndolos en batch de tamaño 1
		if x.ndim == 1:
			x = x[None, :]
		activaciones, _ = self._forward(x)
		return activaciones[-1]

	def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 128, learning_rate: float = 0.01) -> None:
		"""Entrenamiento por descenso de gradiente estocástico simple.

		- Baraja los ejemplos cada época
		- Actualiza por lotes (batch)
		- Imprime accuracy y loss al final de cada época
		"""
		x = np.asarray(x, dtype=np.float32)
		y = np.asarray(y, dtype=np.int64)
		if len(x) != len(y):
			raise ValueError("x e y deben tener la misma longitud")

		for epoch in range(1, epochs + 1):
			perm = np.random.permutation(len(x))
			x_barajado = x[perm]
			y_barajado = y[perm]

			for inicio in range(0, len(x_barajado), batch_size):
				fin = inicio + batch_size
				lote_x = x_barajado[inicio:fin]
				lote_y = y_barajado[inicio:fin]
				self._train_batch(lote_x, lote_y, learning_rate)

			probs = self.predict_proba(x)
			predicciones = np.argmax(probs, axis=1)
			accuracy = float(np.mean(predicciones == y))
			loss = float(self._cross_entropy(probs, y))
			print(f"Epoch {epoch}/{epochs} - accuracy: {accuracy:.4f} - loss: {loss:.4f}")

	def _train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
		"""Actualiza pesos y sesgos usando backpropagation (implementación explícita).

		- `delta` contiene el error en la capa de salida y se propaga hacia atrás
		- Se aplica la derivada de tanh para las capas ocultas
		"""
		activaciones, preactivaciones = self._forward(x)
		probs = activaciones[-1]
		objetivos = np.eye(self.dimensiones[-1], dtype=np.float32)[y]
		delta = (probs - objetivos) / max(len(x), 1)

		for indice in reversed(range(len(self.pesos))):
			entrada = activaciones[indice]
			peso = self.pesos[indice]
			grad_pesos = entrada.T @ delta
			grad_sesgos = np.sum(delta, axis=0)

			# Actualizar parámetros con descenso de gradiente
			self.pesos[indice] = peso - learning_rate * grad_pesos
			self.sesgos[indice] = self.sesgos[indice] - learning_rate * grad_sesgos

			if indice > 0:
				# Propagar delta a la capa anterior aplicando derivada de tanh
				delta = (delta @ peso.T) * (1.0 - np.tanh(preactivaciones[indice - 1]) ** 2)

	@staticmethod
	def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
		# Calcula la pérdida de entropía cruzada promedio.
		indices = np.arange(len(y))
		probabilidades = np.clip(probs[indices, y], 1e-9, 1.0)
		return float(-np.mean(np.log(probabilidades)))

	def save(self, ruta: Path) -> None:
		# Guarda el modelo (dimensiones, pesos y sesgos) a un archivo comprimido .npz.
		contenido: Dict[str, np.ndarray] = {"dimensiones": np.asarray(self.dimensiones, dtype=np.int64)}
		for indice, (peso, sesgo) in enumerate(zip(self.pesos, self.sesgos)):
			contenido[f"peso_{indice}"] = peso
			contenido[f"sesgo_{indice}"] = sesgo
		np.savez_compressed(ruta, **contenido)

	@classmethod
	def load(cls, ruta: Path) -> "RedNeuronalOthello":
		# Carga un modelo desde un fichero .npz y devuelve la instancia reconstruida.
		archivo = np.load(ruta, allow_pickle=False)
		dimensiones = archivo["dimensiones"].astype(int).tolist()
		modelo = cls(dimensiones=dimensiones)
		for indice in range(len(modelo.pesos)):
			modelo.pesos[indice] = archivo[f"peso_{indice}"]
			modelo.sesgos[indice] = archivo[f"sesgo_{indice}"]
		return modelo


def selecciona_movimiento(modelo: RedNeuronalOthello, tablero: Sequence[Sequence[int]], jugador: int):
	"""Selecciona el mejor movimiento según el modelo:

	- Para cada movimiento legal, aplica la jugada en una copia del tablero
	- Predice la probabilidad y elige el movimiento con mayor score.
	En este caso se usa la probabilidad de `LABEL_LOSS` como heurística (convención del autor).
	"""
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


def entrena_modelo(num_partidas: int, epochs: int, batch_size: int, learning_rate: float, ruta_salida: Path) -> RedNeuronalOthello:
	"""Genera datos y entrena un modelo; guarda el resultado en `ruta_salida`."""
	print(f"Generando datos con {num_partidas} partidas aleatorias...")
	x, y = genera_dataset(num_partidas)
	print(f"Ejemplos generados: {len(x)}")

	modelo = RedNeuronalOthello()
	modelo.fit(x, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
	modelo.save(ruta_salida)
	print(f"Modelo guardado en {ruta_salida}")
	return modelo


def juega_contra_modelo(ruta_modelo: Path) -> None:
	"""Carga un modelo y juega una partida completa mostrando movimientos por consola."""
	modelo = RedNeuronalOthello.load(ruta_modelo)
	tablero = nuevo_tablero()
	jugador = 2

	while True:
		movimientos = movimientos_legales(tablero, jugador)
		jugador_oponente = 3 - jugador
		# Si ambos jugadores no tienen movimientos, partida terminada
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
	"""Construye el parser de argumentos para la línea de comandos.

	Comandos soportados:
	- `train`: generar datos y entrenar
	- `play`: cargar modelo y jugar partida de prueba
	"""
	parser = argparse.ArgumentParser(description="Entrena una red neuronal simple para Othello/Reversi.")
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