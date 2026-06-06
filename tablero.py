from __future__ import annotations

from typing import Dict, Optional, Tuple

import pygame

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales, nuevo_tablero

Position = Tuple[int, int]


class TableroUI:
    """Encapsula la parte visual del juego."""

    BG_COLOR = (12, 17, 29)
    GRID_COLOR = (18, 98, 151)
    EMPTY_CELL = (20, 120, 60)
    TEXT_COLOR = (235, 239, 243)
    BLACK_PIECE = (30, 30, 30)
    WHITE_PIECE = (235, 235, 235)
    LEGAL_MOVE_COLOR = (240, 240, 130)

    CELL_SIZE = 80
    HEADER_HEIGHT = 100
    FPS = 60

    def __init__(self) -> None:
        self.board_size = 8
        self.board = nuevo_tablero()

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

        self.current_player = 2
        self.legal_moves: Dict[Position, list[Position]] = {}
        self.game_over = False
        self.message = ""

        self._refresh_game_state()

    def run(self) -> None:
        """Inicializa pygame y ejecuta el bucle principal del juego."""
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        width, height = self.compute_window_size()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Otelo")

        running = True
        while running:
            assert self.clock is not None
            self.clock.tick(self.FPS)
            running = self._handle_events()
            self.draw_scene()

        pygame.quit()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._reset_game()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
        return True

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        if self.game_over:
            return

        board_pos = self._pixel_to_board(pos)
        if board_pos is None:
            return

        if board_pos not in self.legal_moves:
            self.message = "Movimiento no legal"
            return

        aplicar_movimiento(self.board, board_pos, self.current_player, self.legal_moves[board_pos])
        self.current_player = 3 - self.current_player
        self.message = ""
        self._refresh_game_state()

    def _refresh_game_state(self) -> None:
        self.legal_moves = movimientos_legales(self.board, self.current_player)
        if self.legal_moves:
            return

        opponent = 3 - self.current_player
        opponent_moves = movimientos_legales(self.board, opponent)
        if opponent_moves:
            self.current_player = opponent
            self.legal_moves = opponent_moves
            self.message = "Sin movimientos: turno pasado"
            return

        self.game_over = True
        negro, blanco = cuenta_fichas(self.board)
        if negro > blanco:
            resultado = "Gana Negro"
        elif blanco > negro:
            resultado = "Gana Blanco"
        else:
            resultado = "Empate"
        self.message = f"Fin de partida — {resultado}"

    def _reset_game(self) -> None:
        self.board = nuevo_tablero()
        self.current_player = 2
        self.game_over = False
        self.message = ""
        self._refresh_game_state()

    def draw_scene(self) -> None:
        if self.screen is None:
            return

        self.screen.fill(self.BG_COLOR)
        self.draw_header()
        self.draw_board()
        pygame.display.flip()

    def draw_header(self) -> None:
        assert self.screen and self.font

        negro, blanco = cuenta_fichas(self.board)
        turn_text = "Negro (X)" if self.current_player == 2 else "Blanco (O)"
        header_lines = [
            f"Negro: {negro}    Blanco: {blanco}",
            f"Turno actual: {turn_text}",
            "Clic en un movimiento legal para colocar ficha.",
            "Presiona R para reiniciar, ESC para salir.",
        ]

        for index, line in enumerate(header_lines):
            surface = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(surface, (20, 18 + index * 24))

        if self.message:
            message_surface = self.font.render(self.message, True, self.LEGAL_MOVE_COLOR)
            self.screen.blit(message_surface, (20, 18 + len(header_lines) * 24))

    def draw_board(self) -> None:
        assert self.screen is not None

        for row in range(self.board_size):
            for col in range(self.board_size):
                rect = self.cell_rect(row, col)
                pygame.draw.rect(self.screen, self.EMPTY_CELL, rect)
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect, width=2)

                if (row, col) in self.legal_moves and not self.game_over:
                    center = rect.center
                    pygame.draw.circle(self.screen, self.LEGAL_MOVE_COLOR, center, 8)

                piece = self.board[row][col]
                if piece != 0:
                    center = rect.center
                    radius = self.CELL_SIZE // 2 - 10
                    color = self.BLACK_PIECE if piece == 2 else self.WHITE_PIECE
                    pygame.draw.circle(self.screen, color, center, radius)

    def compute_window_size(self) -> Tuple[int, int]:
        width = self.board_size * self.CELL_SIZE
        height = self.board_size * self.CELL_SIZE + self.HEADER_HEIGHT
        return width, height

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = col * self.CELL_SIZE
        y = self.HEADER_HEIGHT + row * self.CELL_SIZE
        return pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

    def _pixel_to_board(self, pos: Tuple[int, int]) -> Optional[Position]:
        x, y = pos
        if y < self.HEADER_HEIGHT:
            return None
        row = (y - self.HEADER_HEIGHT) // self.CELL_SIZE
        col = x // self.CELL_SIZE

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None


if __name__ == "__main__":
    ui = TableroUI()
    ui.run()
