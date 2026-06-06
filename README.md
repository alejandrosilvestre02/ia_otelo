# ia_otelo

Proyecto de Othello/Reversi con un tablero jugable y un entrenamiento sencillo de red neuronal.

## Requisitos

```bash
pip install -r requirements.txt
```

## Entrenar la red

```bash
python entrenamiento.py train --games 300 --epochs 15 --batch-size 128 --learning-rate 0.01
```

El modelo se guarda en `modelo_otelo.npz`.

## Probar el modelo

```bash
python entrenamiento.py play --model modelo_otelo.npz
```

## Jugar en la interfaz

Si existe `modelo_otelo.npz` en la raíz del proyecto, `tablero.py` lo carga y la IA juega automáticamente con las fichas blancas.

```bash
python tablero.py
```

Al arrancar, pulsa `N` para jugar como Negro o `B` para jugar como Blanco.