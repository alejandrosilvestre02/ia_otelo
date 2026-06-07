from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales

Board = List[List[int]]
Move = Tuple[int, int]
Flips = List[Tuple[int, int]]

LABEL_LOSS = 0
LABEL_DRAW = 1
LABEL_WIN = 2


def codifica_tablero(tablero: Sequence[Sequence[int]], jugador: int) -> np.ndarray:
	"""Codifica el tablero como vector plano desde la perspectiva de `jugador`."""
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


@dataclass
class Nodo:
	"""Nodo del árbol MCTS."""

	estado: Board
	jugador: int
	movimiento: Optional[Move] = None
	padre: Optional[Nodo] = None
	n_visitas: int = 0
	valor_total: float = 0.0
	prior: float = 1.0
	hijos: Dict[Optional[Move], Nodo] = field(default_factory=dict)
	movimientos_pendientes: List[Tuple[Optional[Move], Optional[Flips], float]] = field(default_factory=list)

	def __post_init__(self) -> None:
		self.movimientos_pendientes = self._genera_movimientos_pendientes()

	def _genera_movimientos_pendientes(self) -> List[Tuple[Optional[Move], Optional[Flips], float]]:
		movimientos = list(movimientos_legales(self.estado, self.jugador).items())
		if movimientos:
			random.shuffle(movimientos)
			return [(movimiento, flips, 1.0) for movimiento, flips in movimientos]

		# Si el jugador actual no mueve pero el rival sí, se modela un pase.
		if movimientos_legales(self.estado, 3 - self.jugador):
			return [(None, None, 1.0)]
		return []

	def es_terminal(self) -> bool:
		return not movimientos_legales(self.estado, self.jugador) and not movimientos_legales(self.estado, 3 - self.jugador)

	def completamente_expandido(self) -> bool:
		return len(self.movimientos_pendientes) == 0

	def mejor_hijo_final(self) -> Optional[Nodo]:
		if not self.hijos:
			return None
		return max(self.hijos.values(), key=lambda hijo: hijo.n_visitas)

	def mejor_hijo_uct(self, c: float, root_player: int) -> Optional[Nodo]:
		if not self.hijos:
			return None

		def puntuacion(hijo: Nodo) -> float:
			media = hijo.valor_total / hijo.n_visitas if hijo.n_visitas else 0.0
			prior = max(hijo.prior, 1e-6)
			exploracion = c * prior * math.sqrt(max(self.n_visitas, 1)) / (1 + hijo.n_visitas)
			return media + exploracion

		return max(self.hijos.values(), key=puntuacion)


class AgenteUCT:
	"""Agente que usa UCT/MCTS para elegir movimientos en Otelo."""

	def __init__(
		self,
		iteraciones: int = 500,
		c: float = math.sqrt(2.0),
		modelo=None,
		prior_exploration: float = 1.0,
		rollout_limite: int = 60,
	) -> None:
		self.iteraciones = iteraciones
		self.c = c
		self.modelo = modelo
		self.prior_exploration = prior_exploration
		self.rollout_limite = rollout_limite

	def elegir_movimiento(self, tablero: Board, jugador: int) -> Optional[Tuple[Move, Flips]]:
		movimientos = movimientos_legales(tablero, jugador)
		if not movimientos:
			return None
		if len(movimientos) == 1:
			return next(iter(movimientos.items()))

		raiz = Nodo(self._clonar_tablero(tablero), jugador)
		raiz.movimientos_pendientes = self._prioriza_movimientos(raiz.estado, raiz.jugador)
		for _ in range(self.iteraciones):
			nodo = self._seleccionar(raiz)
			if not nodo.es_terminal():
				nodo = self._expandir(nodo)
			recompensa = self._default_policy(nodo.estado, nodo.jugador, jugador)
			self._retropropagar(nodo, recompensa)

		mejor = raiz.mejor_hijo_final()
		if mejor is None or mejor.movimiento is None:
			return None
		return mejor.movimiento, self._flips_desde_movimiento(tablero, mejor.movimiento, jugador)

	def _seleccionar(self, nodo: Nodo) -> Nodo:
		while not nodo.es_terminal() and nodo.completamente_expandido() and nodo.hijos:
			mejor = nodo.mejor_hijo_uct(self.c, nodo.jugador)
			if mejor is None:
				break
			nodo = mejor
		return nodo

	def _expandir(self, nodo: Nodo) -> Nodo:
		if nodo.es_terminal() or not nodo.movimientos_pendientes:
			return nodo

		movimiento, fichas_volteadas, prior = nodo.movimientos_pendientes.pop(0)
		nuevo_estado = self._clonar_tablero(nodo.estado)
		nuevo_jugador = 3 - nodo.jugador
		if movimiento is not None and fichas_volteadas is not None:
			aplicar_movimiento(nuevo_estado, movimiento, nodo.jugador, fichas_volteadas)

		hijo = Nodo(
			estado=nuevo_estado,
			jugador=nuevo_jugador,
			movimiento=movimiento,
			padre=nodo,
			prior=prior,
		)
		hijo.movimientos_pendientes = self._prioriza_movimientos(hijo.estado, hijo.jugador)
		nodo.hijos[movimiento] = hijo
		return hijo

	def _default_policy(self, estado: Board, jugador: int, root_player: int) -> float:
		if self.modelo is not None:
			return self._evalua_con_modelo(estado, jugador, root_player)
		return self._rollout_aleatorio(estado, jugador, root_player)

	def _evalua_con_modelo(self, estado: Board, jugador: int, root_player: int) -> float:
		vector = codifica_tablero(estado, jugador)
		probs = self.modelo.predict_proba(vector)[0]
		valor_actual = float(probs[LABEL_WIN] - probs[LABEL_LOSS])
		return valor_actual if jugador == root_player else -valor_actual

	def _prioriza_movimientos(self, estado: Board, jugador: int) -> List[Tuple[Optional[Move], Optional[Flips], float]]:
		movimientos = list(movimientos_legales(estado, jugador).items())
		if not movimientos:
			if movimientos_legales(estado, 3 - jugador):
				return [(None, None, 1.0)]
			return []

		if self.modelo is None:
			random.shuffle(movimientos)
			prior = 1.0 / len(movimientos)
			return [(movimiento, flips, prior) for movimiento, flips in movimientos]

		scores: List[float] = []
		for movimiento, flips in movimientos:
			nuevo_estado = self._clonar_tablero(estado)
			aplicar_movimiento(nuevo_estado, movimiento, jugador, flips)
			next_player = 3 - jugador
			scores.append(-self._valor_estado_con_modelo(nuevo_estado, next_player))

		priors = self._softmax(np.asarray(scores, dtype=np.float32))
		return [
			(movimiento, flips, float(prior))
			for (movimiento, flips), prior in zip(movimientos, priors)
		]

	def _valor_estado_con_modelo(self, estado: Board, jugador: int) -> float:
		vector = codifica_tablero(estado, jugador)
		probs = self.modelo.predict_proba(vector)[0]
		return float(probs[LABEL_WIN] - probs[LABEL_LOSS])

	@staticmethod
	def _softmax(valores: np.ndarray) -> np.ndarray:
		if valores.size == 0:
			return valores
		desplazado = valores - np.max(valores)
		exp = np.exp(desplazado)
		return exp / np.sum(exp)

	def _rollout_aleatorio(self, estado: Board, jugador: int, root_player: int) -> float:
		copia = self._clonar_tablero(estado)
		turno = jugador
		pasadas = 0
		profundidad = 0

		while pasadas < 2 and profundidad < self.rollout_limite:
			movimientos = movimientos_legales(copia, turno)
			if movimientos:
				pasadas = 0
				movimiento, fichas_volteadas = random.choice(list(movimientos.items()))
				aplicar_movimiento(copia, movimiento, turno, fichas_volteadas)
			else:
				pasadas += 1
			turno = 3 - turno
			profundidad += 1

		negro, blanco = cuenta_fichas(copia)
		if negro == blanco:
			return 0.0
		ganador = 2 if negro > blanco else 1
		return 1.0 if ganador == root_player else -1.0

	def _retropropagar(self, nodo: Nodo, recompensa: float) -> None:
		while nodo is not None:
			nodo.n_visitas += 1
			nodo.valor_total += recompensa
			nodo = nodo.padre

	@staticmethod
	def _clonar_tablero(tablero: Board) -> Board:
		return [fila[:] for fila in tablero]

	@staticmethod
	def _flips_desde_movimiento(tablero: Board, movimiento: Move, jugador: int) -> Flips:
		return list(movimientos_legales(tablero, jugador)[movimiento])



def selecciona_movimiento(
	tablero: Board,
	jugador: int,
	modelo=None,
	iteraciones: int = 500,
	c: float = math.sqrt(2.0),
	rollout_limite: int = 60,
) -> Optional[Tuple[Move, Flips]]:
	"""Atajo funcional para elegir jugada con UCT."""
	agente = AgenteUCT(iteraciones=iteraciones, c=c, modelo=modelo, rollout_limite=rollout_limite)
	return agente.elegir_movimiento(tablero, jugador)
