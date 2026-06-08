"""
Juego de Otelo — implementación básica en Python.

Uso:
    Ejecutar `python otelo.py` y seguir las instrucciones en consola.

Este archivo contiene:
    - Representación del tablero
    - Cálculo de movimientos legales
    - Aplicación de movimientos
    - Lógica de turnos
    - Interfaz de juego por consola
"""
from typing import List, Tuple, Dict
import random

# Tipo para representar el tablero:
# 0 = vacío, 1 = ficha blanca, 2 = ficha negra
Board = List[List[int]]

# Direcciones posibles para capturar fichas en Otelo (8 direcciones)
DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
]


# ---------------------------------------------------------
#  Inicialización del tablero
# ---------------------------------------------------------

def nuevo_tablero() -> Board:
    """
    Crea un tablero inicial de Otelo (8x8) con la configuración estándar:
        - Blancas en (3,3) y (4,4)
        - Negras en (3,4) y (4,3)
    """
    b = [[0] * 8 for _ in range(8)]
    b[3][3] = 1
    b[3][4] = 2
    b[4][3] = 2
    b[4][4] = 1
    return b


def dentro(i: int, j: int) -> bool:
    """Devuelve True si la posición (i,j) está dentro del tablero."""
    return 0 <= i < 8 and 0 <= j < 8


# ---------------------------------------------------------
#  Cálculo de movimientos legales
# ---------------------------------------------------------

def movimientos_legales(b: Board, jugador: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Calcula todos los movimientos legales para un jugador en Otelo.

    Devuelve un diccionario:
        (fila, columna) -> lista de fichas que serían volteadas

    Un movimiento es legal si:
        - Se coloca en una casilla vacía
        - En alguna dirección hay una o más fichas del rival
        - Y al final de esa línea hay una ficha propia
    """
    moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    opponent = 3 - jugador

    for i in range(8):
        for j in range(8):
            if b[i][j] != 0:
                continue  # Solo se puede jugar en casillas vacías

            flips: List[Tuple[int, int]] = []

            # Explorar las 8 direcciones
            for di, dj in DIRS:
                ni, nj = i + di, j + dj
                line: List[Tuple[int, int]] = []

                # Avanzar mientras haya fichas del oponente
                while dentro(ni, nj) and b[ni][nj] == opponent:
                    line.append((ni, nj))
                    ni += di
                    nj += dj

                # Si la línea termina en una ficha propia, es un movimiento válido
                if line and dentro(ni, nj) and b[ni][nj] == jugador:
                    flips.extend(line)

            if flips:
                moves[(i, j)] = flips

    return moves


# ---------------------------------------------------------
#  Aplicación de movimientos
# ---------------------------------------------------------

def aplicar_movimiento(b: Board, move: Tuple[int, int], jugador: int, flips: List[Tuple[int, int]]) -> None:
    """
    Coloca una ficha del jugador en 'move' y voltea las fichas indicadas.
    """
    i, j = move
    b[i][j] = jugador
    for x, y in flips:
        b[x][y] = jugador


# ---------------------------------------------------------
#  Utilidades del juego
# ---------------------------------------------------------

def cuenta_fichas(b: Board) -> Tuple[int, int]:
    """
    Cuenta cuántas fichas negras y blancas hay en el tablero.
    Devuelve (negras, blancas).
    """
    negro = sum(1 for row in b for v in row if v == 2)
    blanco = sum(1 for row in b for v in row if v == 1)
    return negro, blanco


def mostrar_tablero(b: Board, legal_moves: set = None) -> None:
    """
    Muestra el tablero en consola con formato visual.
    Si se pasa 'legal_moves', se podrían resaltar movimientos legales.
    (Actualmente solo se muestran, no se resaltan visualmente.)
    """
    letras = 'abcdefgh'
    max_row_digits = len(str(8))
    prefix_width = max_row_digits + 2

    print(' ' * prefix_width + '   '.join(letras))

    for i in range(8):
        fila = []
        for j in range(8):
            v = b[i][j]
            if v == 2:
                fila.append('X')  # Negro
            elif v == 1:
                fila.append('O')  # Blanco
            else:
                fila.append(' ')
        row_str = '|'.join(f' {c} ' for c in fila)
        line = f"{i+1} |" + row_str + '|'
        sep = '-' * len(line)
        print(sep)
        print(line)

    print(sep)


def parsea_entrada(s: str) -> Tuple[int, int]:
    """
    Convierte una entrada del usuario (ej: 'd3' o '3 4') en coordenadas (fila, columna).
    Acepta:
        - formato tipo 'd3'
        - formato tipo '3 4'
    """
    s = s.strip().lower()

    # Formato tipo 'd3'
    if len(s) >= 2 and s[0].isalpha():
        col = ord(s[0]) - ord('a')
        try:
            row = int(s[1:]) - 1
        except ValueError:
            raise ValueError('Formato incorrecto')
        return row, col

    # Formato tipo '3 4'
    parts = s.split()
    if len(parts) == 2:
        return int(parts[0]) - 1, int(parts[1]) - 1

    raise ValueError('Formato de entrada no reconocido')


def movimiento_aleatorio(b: Board, jugador: int):
    """
    Devuelve un movimiento aleatorio entre los legales.
    Si no hay movimientos, devuelve None.
    """
    moves = list(movimientos_legales(b, jugador).items())
    if not moves:
        return None
    return random.choice(moves)


# ---------------------------------------------------------
#  Interfaz de juego por consola
# ---------------------------------------------------------

def jugar_cli():
    """
    Permite jugar una partida completa de Otelo desde la consola.
    El jugador negro (X) empieza.
    """
    b = nuevo_tablero()
    jugador = 2  # 2 = negro, 1 = blanco
    nombres = {2: 'Negro (X)', 1: 'Blanco (O)'}

    while True:
        opponent = 3 - jugador
        moves = movimientos_legales(b, jugador)
        moves_opp = movimientos_legales(b, opponent)

        mostrar_tablero(b, set(moves.keys()))
        negro, blanco = cuenta_fichas(b)
        print(f'Puntuación — Negro: {negro}  Blanco: {blanco}')

        # Fin de partida
        if not moves and not moves_opp:
            print('No hay más movimientos disponibles. Fin de la partida.')
            break

        # Si el jugador no puede mover, pasa turno
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


# ---------------------------------------------------------
#  Ejecución principal
# ---------------------------------------------------------

if __name__ == '__main__':
    print('Otelo — juego básico')
    jugar_cli()
