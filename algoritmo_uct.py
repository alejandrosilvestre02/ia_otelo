import math
import random

tree = {}
"""return the move in ACTIONS(state) whose node has highest number of playouts
"""

class Nodo:
    """
    Representa un nodo en el árbol MCTS.

    Atributos
    ---------
    estado      : Othello  - copia del estado del juego en este nodo
    jugador     : int      - jugador que acaba de mover para llegar aquí
    movimiento  : tuple    - jugada que generó este nodo desde el padre
    padre       : NodoUCT  - nodo padre (None para la raíz)
    hijos       : list     - lista de nodos hijo expandidos
    n_visitas   : int      - número de veces que este nodo ha sido visitado
    valor       : float    - valor acumulado (suma de recompensas)
    movs_sin_expandir : list - movimientos del estado aún no explorados
    """
    def __init__(self, estado, jugador, movimiento=None, padre=None):
        self.estado = estado
        self.jugador = jugador
        self.movimiento = movimiento
        self.padre = padre
        self.hijos = []
        """pag 208, 27/35 -> n_victorias / n_visitas"""
        self.n_visitas = 0 
        self.n_victorias = 0.0
        
        # Movimientos pendientes de expandir en este nodo
        turno_actual = estado.turno
        if turno_actual is not None:
            self.movs_sin_expandir = estado.movimientos_validos(turno_actual)[:]
            random.shuffle(self.movs_sin_expandir)
        else:
            self.movs_sin_expandir = []

def nodo_terminal(self):
    return self.estado.ha_terminado()

def nodo_completamente_expandido(self):
    return len(self.movs_sin_expandir) == 0

def best_child(self, c=math.squrt(2)):
    """UCB1 = (valor/visitas) + c * sqrt(ln(N) / n)"""

    def ucb(hijo):
        if hijo.n_visitas == 0:
            return float('inf')
        explotacion = hijo.n_victorias / hijo.n_visitas
        exploracion = c * math.sqrt(math.log(self.n_visitas) / hijo.n_visitas)
        return explotacion + exploracion

    return max(self.hijos, key=ucb)

def hijo_mas_visitado_final(self):
    """elige al hijo más visitado cuando se terminan las iteraciones"""
    return max(self.hijos, key=lambda h: h.n_visitas)

def tree_policy(self, nodo):
    """
        TREE POLICY: desciende por el árbol usando UCB1 hasta encontrar
        un nodo no completamente expandido o terminal.
        """
    while not nodo.es_terminal():
        if not nodo.esta_completamente_expandido():
            return expand(nodo)
        else:
            nodo = nodo.best_child(self.c)
    return nodo

def default_policy(estado):
    while not estado.is_terminal():
        accion = random.choice(estado.get_valid_actions())
        estado = estado.apply_action(accion)
    return estado.get_reward()

def expand(self, nodo):

    #EXPANSION: toma un movimiento aún no explorado y crea un hijo.

    if nodo.es_terminal() or not nodo.movs_sin_expandir: 
        return nodo
    
    movimiento = nodo.movs_sin_expandir.pop()
    nuevo_estado = nodo.estado.clonar()
    nuevo_estado.aplicar_movimiento(*movimiento)

# El jugador del nuevo nodo es quien acaba de mover
    hijo = Nodo(
            estado=nuevo_estado,
            jugador=nodo.estado.turno,   # jugador que acaba de mover
            movimiento=movimiento,
            padre=nodo
        )
    nodo.hijos.append(hijo)
    return hijo

def backup(self, nodo, recompensa, jugador_activo):
    while nodo is not None:
        nodo.n_visitas += 1
        if nodo.jugador == jugador_activo:
            nodo.valor += recompensa
        else:
            nodo.valor -= recompensa
        nodo = nodo.padre