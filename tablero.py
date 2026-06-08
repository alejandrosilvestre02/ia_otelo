from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple

import pygame

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales, nuevo_tablero
from algoritmo_uct import selecciona_movimiento
from entrenamiento import DEFAULT_MODEL_PATH, RedNeuronalOtelo

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
    PANEL_COLOR = (22, 30, 48)
    PANEL_BORDER = (77, 132, 191)
    BUTTON_COLOR = (36, 124, 76)
    BUTTON_HOVER_COLOR = (46, 150, 92)
    BUTTON_TEXT_COLOR = (245, 247, 249)
    AI_SEARCH_ITERATIONS = 30
    AI_ROLLOUT_LIMIT = 15
    AI_MOVE_DELAY_MS = 0


    CELL_SIZE = 80
    HEADER_HEIGHT = 100
    FPS = 60
    HUMAN_PLAYER = 2
    AI_PLAYER = 1

    def __init__(self) -> None:
        self.board_size = 8
        self.board = nuevo_tablero()

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

        self.human_player = self.HUMAN_PLAYER
        self.ai_player = self.AI_PLAYER
        self.current_player = 2
        self.legal_moves: Dict[Position, list[Position]] = {}
        self.game_over = False
        self.model_path = DEFAULT_MODEL_PATH
        self.message = "Pulsa N para jugar como Negro o B para jugar como Blanco."
        self.model: Optional[RedNeuronalOtelo] = self.load_model(self.model_path)
        self.awaiting_selection = True
        self.start_button_black: Optional[pygame.Rect] = None
        self.start_button_white: Optional[pygame.Rect] = None
        self.pending_ai_move_at: Optional[int] = None
        self.ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="otelo-ai")
        self.ai_future: Optional[concurrent.futures.Future[Optional[Tuple[Position, list[Position]]]]] = None
        self.ai_request_id = 0
        self.ai_future_request_id = 0
        self.ai_thinking = False

        self.refresh_game_state()

    def run(self) -> None:
        """Inicializa pygame y ejecuta el bucle principal del juego."""
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        width, height = self.compute_window_size()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Otelo")

        try:
            running = True
            while running:
                assert self.clock is not None
                self.clock.tick(self.FPS)
                running = self.handle_events()
                self.update_pending_ai_turn()
                self.draw_scene()
        finally:
            self.ai_executor.shutdown(wait=False, cancel_futures=True)
            pygame.quit()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset_game()
                if self.awaiting_selection:
                    if event.key == pygame.K_n:
                        self.start_match(human_player=2)
                    elif event.key == pygame.K_b:
                        self.start_match(human_player=1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
        return True

    def handle_click(self, pos: Tuple[int, int]) -> None:
        if self.awaiting_selection:
            if self.handle_start_screen_click(pos):
                return
            return

        if self.game_over or self.awaiting_selection or self.current_player != self.human_player:
            return

        board_pos = self.pixel_to_board(pos)
        if board_pos is None:
            return

        if board_pos not in self.legal_moves:
            self.message = "Movimiento no legal"
            return

        aplicar_movimiento(self.board, board_pos, self.current_player, self.legal_moves[board_pos])
        self.current_player = 3 - self.current_player
        self.message = ""
        self.refresh_game_state()
        self.schedule_ai_turn_if_needed()

    def refresh_game_state(self) -> None:
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

    def schedule_ai_turn_if_needed(self) -> None:
        if self.game_over or self.model is None or self.current_player != self.ai_player:
            self.pending_ai_move_at = None
            self.ai_thinking = False
            return

        if self.ai_future is not None:
            self.ai_thinking = True
            return

        self.pending_ai_move_at = pygame.time.get_ticks() + self.AI_MOVE_DELAY_MS

    def update_pending_ai_turn(self) -> None:
        if self.ai_future is not None and self.ai_future.done():
            future = self.ai_future
            request_id = self.ai_future_request_id
            self.ai_future = None
            self.ai_thinking = False

            if request_id != self.ai_request_id or self.game_over or self.current_player != self.ai_player:
                return

            try:
                seleccion = future.result()
            except Exception as exc:  # pragma: no cover - defensivo
                self.message = f"Error calculando la jugada de la IA: {exc}"
                return

            if seleccion is None:
                self.refresh_game_state()
                self.schedule_ai_turn_if_needed()
                return

            movimiento, fichas_volteadas = seleccion
            aplicar_movimiento(self.board, movimiento, self.current_player, fichas_volteadas)
            self.current_player = 3 - self.current_player
            self.message = "La IA ha jugado"
            self.refresh_game_state()
            self.schedule_ai_turn_if_needed()
            return

        if self.pending_ai_move_at is None:
            return

        if pygame.time.get_ticks() < self.pending_ai_move_at:
            return

        self.pending_ai_move_at = None
        self.start_ai_search()

    def start_ai_search(self) -> None:
        if self.game_over or self.model is None or self.current_player != self.ai_player:
            return

        if self.ai_future is not None:
            self.ai_thinking = True
            return

        self.ai_request_id += 1
        self.ai_future_request_id = self.ai_request_id
        tablero_snapshot = [fila[:] for fila in self.board]
        jugador = self.current_player
        modelo = self.model
        self.ai_thinking = True
        self.message = "La IA está pensando..."
        self.ai_future = self.ai_executor.submit(
            selecciona_movimiento,
            tablero_snapshot,
            jugador,
            modelo,
            self.AI_SEARCH_ITERATIONS,
            1.4,
            self.AI_ROLLOUT_LIMIT,
        )

    def reset_game(self) -> None:
        self.board = nuevo_tablero()
        self.current_player = 2
        self.game_over = False
        self.message = ""
        self.pending_ai_move_at = None
        self.ai_future = None
        self.ai_request_id += 1
        self.ai_thinking = False
        self.refresh_game_state()

        if self.model is not None and self.current_player == self.ai_player:
            self.schedule_ai_turn_if_needed()

    def start_match(self, human_player: int) -> None:
        self.human_player = human_player
        self.ai_player = 3 - human_player
        self.awaiting_selection = False
        self.reset_game()

    def handle_start_screen_click(self, pos: Tuple[int, int]) -> bool:
        if self.start_button_black and self.start_button_black.collidepoint(pos):
            self.start_match(human_player=2)
            return True

        if self.start_button_white and self.start_button_white.collidepoint(pos):
            self.start_match(human_player=1)
            return True

        return False

    def load_model(self, ruta: Path) -> Optional[RedNeuronalOtelo]:
        if not ruta.exists():
            return RedNeuronalOtelo()

        try:
            return RedNeuronalOtelo.load(ruta)
        except Exception as exc:  # pragma: no cover - defensivo para errores de carga
            self.message = f"No se pudo cargar el modelo: {exc}"
            return RedNeuronalOtelo()

    def draw_scene(self) -> None:
        if self.screen is None:
            return

        self.screen.fill(self.BG_COLOR)
        if self.awaiting_selection:
            self.draw_start_screen()
            pygame.display.flip()
            return

        self.draw_header()
        self.draw_board()
        pygame.display.flip()

    def draw_start_screen(self) -> None:
        assert self.screen and self.font

        width, height = self.compute_window_size()
        panel_width = 520
        panel_height = 290
        panel_rect = pygame.Rect(
            (width - panel_width) // 2,
            (height - panel_height) // 2,
            panel_width,
            panel_height,
        )
        pygame.draw.rect(self.screen, self.PANEL_COLOR, panel_rect, border_radius=18)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, panel_rect, width=3, border_radius=18)

        title_font = pygame.font.Font(None, 42)
        title = title_font.render("Othello / Reversi", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.centerx - title.get_width() // 2, panel_rect.top + 28))

        lines = [
            "Elige cómo quieres jugar.",
            "Haz clic en una opción para empezar.",
            "También puedes pulsar N o B si la ventana tiene el foco.",
        ]
        for index, line in enumerate(lines):
            surface = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(surface, (panel_rect.left + 32, panel_rect.top + 92 + index * 28))

        button_width = 180
        button_height = 54
        gap = 24
        button_y = panel_rect.bottom - 86
        black_rect = pygame.Rect(panel_rect.left + 58, button_y, button_width, button_height)
        white_rect = pygame.Rect(black_rect.right + gap, button_y, button_width, button_height)
        self.start_button_black = black_rect
        self.start_button_white = white_rect

        mouse_pos = pygame.mouse.get_pos()
        for rect, label, detail in (
            (black_rect, "Jugar Negro", "IA blanca"),
            (white_rect, "Jugar Blanco", "IA negra"),
        ):
            color = self.BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else self.BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=12)
            pygame.draw.rect(self.screen, self.PANEL_BORDER, rect, width=2, border_radius=12)

            label_surface = self.font.render(label, True, self.BUTTON_TEXT_COLOR)
            detail_surface = self.font.render(detail, True, self.BUTTON_TEXT_COLOR)
            self.screen.blit(
                label_surface,
                (rect.centerx - label_surface.get_width() // 2, rect.top + 7),
            )
            self.screen.blit(
                detail_surface,
                (rect.centerx - detail_surface.get_width() // 2, rect.top + 29),
            )

    def draw_header(self) -> None:
        assert self.screen and self.font

        negro, blanco = cuenta_fichas(self.board)
        if self.awaiting_selection:
            turn_text = "Selecciona color"
        elif self.current_player == self.human_player:
            turn_text = "Tu turno"
        elif self.ai_thinking or self.ai_future is not None:
            turn_text = "La IA está pensando"
        elif self.model is not None:
            turn_text = "Turno de la IA"
        else:
            turn_text = "Turno actual"
        header_lines = [
            f"Negro: {negro}    Blanco: {blanco}",
            f"Turno actual: {turn_text}",
            "Clic en un movimiento legal para colocar ficha.",
            "Presiona R para reiniciar, ESC para salir.",
        ]

        if self.awaiting_selection:
            header_lines = [
                "Selecciona color antes de empezar.",
                "Pulsa N para jugar como Negro (X) o B para jugar como Blanco (O).",
                "Si existe modelo_otelo.npz, la IA tomará el color contrario.",
                "Presiona ESC para salir.",
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

    def pixel_to_board(self, pos: Tuple[int, int]) -> Optional[Position]:
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
    try:
        ui.run()
    except KeyboardInterrupt:
        pygame.quit()
