import math
import random

from otelo import aplicar_movimiento, cuenta_fichas, movimientos_legales, nuevo_tablero
from entrenamiento import codifica_tablero, LABEL_WIN, LABEL_LOSS


class EstadoOthello:
    """Envuelve el tablero de listas para que AgenteUCT pueda operar sobre él."""

    def __init__(self, tablero, turno):
        self.tablero = [fila[:] for fila in tablero]
        self.turno = turno

    def clonar(self):
        return EstadoOthello(self.tablero, self.turno)

    def movimientos_validos(self, jugador):
        return list(movimientos_legales(self.tablero, jugador).items())

    def aplicar_movimiento(self, movimiento, fichas_volteadas):
        aplicar_movimiento(self.tablero, movimiento, self.turno, fichas_volteadas)
        # Cambiar turno: si el rival tiene movimientos válidos, se le pasa el turno.
        # Si no, el mismo jugador continúa; si ninguno puede mover, la partida termina.
        siguiente = 3 - self.turno
        if movimientos_legales(self.tablero, siguiente):
            self.turno = siguiente
        elif not movimientos_legales(self.tablero, self.turno):
            self.turno = None  # partida terminada

    def ha_terminado(self):
        if self.turno is None:
            return True
        if movimientos_legales(self.tablero, self.turno):
            return False
        if movimientos_legales(self.tablero, 3 - self.turno):
            return False
        return True

    def ganador(self):
        negro, blanco = cuenta_fichas(self.tablero)
        if negro > blanco:
            return 2
        if blanco > negro:
            return 1
        return None


class Nodo:
    """
    Representa un nodo en el árbol MCTS.

    Atributos
    ---------
    estado      : EstadoOthello - copia del estado del juego en este nodo
    jugador     : int           - jugador que acaba de mover para llegar aquí
    movimiento  : tuple         - jugada que generó este nodo desde el padre
    padre       : Nodo          - nodo padre (None para la raíz)
    hijos       : list          - lista de nodos hijo expandidos
    n_visitas   : int           - número de veces que este nodo ha sido visitado
    n_victorias : float         - valor acumulado (suma de recompensas)
    movs_sin_expandir : list    - movimientos del estado aún no explorados
    """

    def __init__(self, estado, jugador, movimiento=None, padre=None):
        self.estado = estado
        self.jugador = jugador
        self.movimiento = movimiento
        self.padre = padre
        self.hijos = []
        self.n_visitas = 0
        self.n_victorias = 0.0

        turno_actual = estado.turno
        if turno_actual is not None:
            self.movs_sin_expandir = estado.movimientos_validos(turno_actual)[:]
            random.shuffle(self.movs_sin_expandir)
        else:
            self.movs_sin_expandir = []

    def es_terminal(self):
        return self.estado.ha_terminado()

    def esta_completamente_expandido(self):
        return len(self.movs_sin_expandir) == 0

    def best_child(self, c=math.sqrt(2)):
        """UCB1 = (victorias/visitas) + c * sqrt(ln(N) / n)"""
        def ucb(hijo):
            if hijo.n_visitas == 0:
                return float('inf')
            explotacion = hijo.n_victorias / hijo.n_visitas
            exploracion = c * math.sqrt(math.log(self.n_visitas) / hijo.n_visitas)
            return explotacion + exploracion
        return max(self.hijos, key=ucb)

    def mejor_hijo_final(self, criterio):
        """Elige al hijo más visitado cuando se terminan las iteraciones."""
        if criterio == "max":
            return max(self.hijos, key=lambda h: h.n_victorias / h.n_visitas if h.n_visitas > 0 else float('-inf'))
        elif criterio == "robust":
            return max(self.hijos, key=lambda h: h.n_visitas)
        elif criterio == "max-robust":
            # Se selecciona primero por mayor número de visitas (robustez)
            # y entre estos el de mejor valor medio.
            max_visitas = max(h.n_visitas for h in self.hijos)
            max_valor = max(h.n_victorias / h.n_visitas if h.n_visitas > 0 else float('-inf') for h in self.hijos)
            candidatos = [h for h in self.hijos
                        if h.n_visitas == max_visitas
                        and h.n_visitas > 0
                        and h.n_victorias / h.n_visitas == max_valor]
            if candidatos:
                return candidatos[0]
            return max(self.hijos, key=lambda h: h.n_visitas)
        elif criterio == "secure":
            # Usa la cota inferior de UCB para escoger el hijo más seguro
            # con mayor valor mínimo plausible entre los explorados.
            def lower_bound(h):
                if h.n_visitas == 0:
                    return float('-inf')
                return h.n_victorias / h.n_visitas - math.sqrt(2) * math.sqrt(math.log(self.n_visitas) / h.n_visitas)
            return max(self.hijos, key=lower_bound)


class AgenteUCT:
    """
    Agente que usa el algoritmo UCT para elegir movimientos en Otelo.

    Parámetros
    ----------
    iteraciones : int    - número de simulaciones por jugada
    c           : float  - constante de exploración UCB1 (√2 por defecto)
    red         : objeto con método predecir(tablero) → float ∈ [-1,1]
                         Si es None, se usa rollout aleatorio.
    """

    def __init__(self, iteraciones=500, c=math.sqrt(2), red=None, rollout_limite=60, criterio="max"):
        self.iteraciones = iteraciones
        self.c = c
        self.red = red
        self.rollout_limite = rollout_limite
        self.criterio = criterio

    def elegir_movimiento(self, estado):
        """Devuelve el mejor movimiento (fila, col) para el jugador del turno."""
        jugador = estado.turno
        movs = estado.movimientos_validos(jugador)

        if not movs:
            return None
        if len(movs) == 1:
            return movs[0][0]

        raiz = Nodo(estado.clonar(), jugador)

        for _ in range(self.iteraciones):
            nodo = self.tree_policy(raiz)
            recompensa = self.default_policy(nodo)
            self.backup(nodo, recompensa, jugador)

        return raiz.mejor_hijo_final(criterio=self.criterio).movimiento

    def tree_policy(self, nodo):
        """TREE POLICY: desciende por el árbol usando UCB1 hasta encontrar
        un nodo no completamente expandido o terminal."""
        while not nodo.es_terminal():
            if not nodo.esta_completamente_expandido():
                return self.expand(nodo)
            else:
                nodo = nodo.best_child(self.c)
        return nodo

    def default_policy(self, nodo):
        """DEFAULT POLICY: estima la recompensa desde el nodo dado.

        Si hay red neuronal disponible, se usa para predecir el valor
        del estado actual (evitando la simulación completa).
        Si no, se realiza un rollout aleatorio hasta el final.

        Devuelve un valor en [-1, 1]:
           +1  victoria del jugador raíz
            0  empate
           -1  derrota del jugador raíz
        """
        if self.red is not None:
            # La red predice desde el punto de vista del jugador activo en el nodo
            entrada = codifica_tablero(nodo.estado.tablero, nodo.jugador)
            probs = self.red.predict_proba(entrada)[0]
            # Convertir a escalar en [-1, 1]: P(WIN) - P(LOSS)
            return float(probs[LABEL_WIN] - probs[LABEL_LOSS])

        # --- Rollout aleatorio ---
        # Si no hay red, se simula una partida desde el nodo hasta el final
        # eligiendo movimientos al azar.
        estado_sim = nodo.estado.clonar()
        while not estado_sim.ha_terminado():
            jugador_sim = estado_sim.turno
            movs = estado_sim.movimientos_validos(jugador_sim)
            if movs:
                mov, fichas = random.choice(movs)
                estado_sim.aplicar_movimiento(mov, fichas)
            else:
                break

        ganador = estado_sim.ganador()
        # Si no hay ganador, se considera empate y la recompensa es neutra.
        if ganador is None:
            return 0.0
        # Si el ganador coincide con el jugador que originó este nodo,
        # la simulación es buena para él; de lo contrario, es mala.
        return 1.0 if ganador == nodo.jugador else -1.0

    def expand(self, nodo):
        """EXPANSION: toma un movimiento aún no explorado y crea un hijo."""
        if nodo.es_terminal() or not nodo.movs_sin_expandir:
            return nodo

        movimiento, fichas_volteadas = nodo.movs_sin_expandir.pop()
        nuevo_estado = nodo.estado.clonar()
        nuevo_estado.aplicar_movimiento(movimiento, fichas_volteadas)

        # El jugador del nuevo nodo es quien acaba de mover
        hijo = Nodo(
            estado=nuevo_estado,
            jugador=nodo.estado.turno,  # jugador que acaba de mover
            movimiento=movimiento,
            padre=nodo,
        )
        nodo.hijos.append(hijo)
        return hijo

    def backup(self, nodo, recompensa, jugador_activo):
        """BACKUP: propaga la recompensa hacia arriba por el árbol."""
        while nodo is not None:
            nodo.n_visitas += 1
            if nodo.jugador == jugador_activo:
                nodo.n_victorias += recompensa
            else:
                nodo.n_victorias -= recompensa
            nodo = nodo.padre


def selecciona_movimiento(tablero, jugador, modelo=None, iteraciones=200, c=math.sqrt(2), rollout_limite=60, criterio="max"):
    """Interfaz para tablero.py: usa AgenteUCT y devuelve (movimiento, fichas_volteadas)."""
    from otelo import movimientos_legales as ml
    movimientos = ml(tablero, jugador)
    if not movimientos:
        return None

    estado = EstadoOthello(tablero, jugador)
    agente = AgenteUCT(iteraciones=iteraciones, c=c, red=modelo, rollout_limite=rollout_limite, criterio=criterio)
    movimiento = agente.elegir_movimiento(estado)
    if movimiento is None:
        return None

    # elegir_movimiento devuelve solo (fila, col); recuperamos las fichas del dict
    fichas_volteadas = movimientos[movimiento]
    return movimiento, fichas_volteadas
