# ia_otelo

Proyecto de Otelo con un tablero jugable y un entrenamiento sencillo de red neuronal.

## Requisitos

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install tensorflow numpy pygame
```

El proyecto está pensado para usar el entorno `.venv`, que incluye TensorFlow compatible con Python 3.13.

## Entrenar la red

```bash
python entrenamiento.py train --games 300 --epochs 15 --batch-size 128 --learning-rate 0.01
```

El modelo se guarda en `modelo_otelo.keras`.

## Probar el modelo

```bash
python entrenamiento.py play --model modelo_otelo.keras
```

## Jugar en la interfaz

Si existe `modelo_otelo.keras` en la raíz del proyecto, `tablero.py` lo carga y la IA juega automáticamente con las fichas blancas.

```bash
python tablero.py
```

Al arrancar, pulsa `N` para jugar como Negro o `B` para jugar como Blanco.