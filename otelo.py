"""Juego Othello / Reversi - implementación básica y CLI en español.

Uso: ejecutar `python otelo.py` y seguir las instrucciones.
"""

from typing import List, Tuple, Dict
import random

Board = List[List[int]]  # 0 vacío, 1 blanca, 2 negra

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def nuevo_tablero() -> Board:
	b = [[0] * 8 for _ in range(8)]
	b[3][3] = 1
	b[3][4] = 2
	b[4][3] = 2
	b[4][4] = 1
	return b


def dentro(i: int, j: int) -> bool:
	return 0 <= i < 8 and 0 <= j < 8


def movimientos_legales(b: Board, jugador: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
	moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
	opponent = 3 - jugador
	for i in range(8):
		for j in range(8):
			if b[i][j] != 0:
				continue
			flips: List[Tuple[int, int]] = []
			for di, dj in DIRS:
				ni, nj = i + di, j + dj
				line: List[Tuple[int, int]] = []
				while dentro(ni, nj) and b[ni][nj] == opponent:
					line.append((ni, nj))
					ni += di
					nj += dj
				if line and dentro(ni, nj) and b[ni][nj] == jugador:
					flips.extend(line)
			if flips:
				moves[(i, j)] = flips
	return moves


def aplicar_movimiento(b: Board, move: Tuple[int, int], jugador: int, flips: List[Tuple[int, int]]) -> None:
	i, j = move
	b[i][j] = jugador
	for x, y in flips:
		b[x][y] = jugador


def cuenta_fichas(b: Board) -> Tuple[int, int]:
	negro = sum(1 for row in b for v in row if v == 2)
	blanco = sum(1 for row in b for v in row if v == 1)
	return negro, blanco


def mostrar_tablero(b: Board, legal_moves: set = None) -> None:
	# Muestra el tablero
	letras = 'abcdefgh'
	# encabezado de columnas (alineado según ancho de prefijo de filas)
	max_row_digits = len(str(8))
	prefix_width = max_row_digits + 2  # e.g. '1 |' -> 3 chars
	print(' ' * prefix_width + '   '.join(letras))
	for i in range(8):
		# construir fila con separadores verticales
		fila = []
		for j in range(8):
			v = b[i][j]
			if v == 2:
				fila.append('X')
			elif v == 1:
				fila.append('O')
			else:
				fila.append(' ')
		row_str = '|'.join(f' {c} ' for c in fila)
		line = f"{i+1} |" + row_str + '|'
		# crear un separador con guiones '-' ajustado a la longitud de la línea
		sep = '-' * len(line)
		print(sep)
		print(line)
	# último separador
	print(sep)


def parsea_entrada(s: str) -> Tuple[int, int]:
	s = s.strip().lower()
	if len(s) >= 2 and s[0].isalpha():
		col = ord(s[0]) - ord('a')
		try:
			row = int(s[1:]) - 1
		except ValueError:
			raise ValueError('Formato incorrecto')
		return row, col
	parts = s.split()
	if len(parts) == 2:
		return int(parts[0]) - 1, int(parts[1]) - 1
	raise ValueError('Formato de entrada no reconocido')


def movimiento_aleatorio(b: Board, jugador: int):
	moves = list(movimientos_legales(b, jugador).items())
	if not moves:
		return None
	return random.choice(moves)


def jugar_cli():
	b = nuevo_tablero()
	jugador = 2  # 2 negro, 1 blanco
	nombres = {2: 'Negro (X)', 1: 'Blanco (O)'}
	while True:
		opponent = 3 - jugador
		moves = movimientos_legales(b, jugador)
		moves_opp = movimientos_legales(b, opponent)
		# Mostrar tablero con indicación de movimientos legales
		mostrar_tablero(b, set(moves.keys()))
		negro, blanco = cuenta_fichas(b)
		print(f'Puntuación — Negro: {negro}  Blanco: {blanco}')
		if not moves and not moves_opp:
			print('No hay más movimientos disponibles. Fin de la partida.')
			break

		if not moves:
			print(f'{nombres[jugador]} no tiene movimientos legales y pasa.')
			jugador = opponent
			continue

		print(f'Turno de {nombres[jugador]}')
		print('Movimientos legales: ' + ', '.join(f"{chr(c+97)}{r+1}" for (r, c) in moves.keys()))
		entrada = input('Introduce movimiento (ej: d3): ').strip()

		try:
			move = parsea_entrada(entrada)
		except ValueError:
			print('Entrada inválida. Intenta de nuevo.')
			continue
		if move not in moves:
			print('Movimiento no legal. Intenta de nuevo.')
			continue
		aplicar_movimiento(b, move, jugador, moves[move])
		jugador = opponent

	mostrar_tablero(b)
	negro, blanco = cuenta_fichas(b)
	print(f'Resultado final — Negro: {negro}  Blanco: {blanco}')
	if negro > blanco:
		print('Gana Negro (X)')
	elif blanco > negro:
		print('Gana Blanco (O)')
	else:
		print('Empate')


if __name__ == '__main__':
	print('Otelo - juego basico')
	jugar_cli()

