"""
Genera el conjunto de entrenamiento para el detector de fugas.

Produce secuencias etiquetadas automáticamente: como el penacho lo
creamos nosotros, sabemos con exactitud dónde está en cada fotograma.
Eso evita tener que etiquetar video real a mano, que es el cuello de
botella de este tipo de proyectos.

Salida en formato YOLO:
    dataset/images/train/*.png
    dataset/labels/train/*.txt      clase cx cy w h   (normalizado)
"""

import os
import json
import numpy as np
import cv2
from fondo import escena_ogi, aplicar_temblor, perturbar
from penacho import Penacho, componer, caja


def generar_secuencia(alto=256, ancho=448, n_frames=12, con_fuga=True,
                      semilla=None, fps=6.0):
    """
    Devuelve (frames, cajas). Cuando no hay fuga, las cajas van vacías:
    esos ejemplos negativos son necesarios para que el detector no
    invente penachos donde solo hay temblor de cámara o vapor.
    """
    rng = np.random.default_rng(semilla)
    fondo, accesorios = escena_ogi(alto, ancho, semilla=semilla)

    penachos = []
    if con_fuga:
        for _ in range(rng.integers(1, 3)):
            ox, oy = accesorios[rng.integers(0, len(accesorios))]
            penachos.append(Penacho(
                alto, ancho, origen=(ox + rng.integers(-3, 4), oy),
                intensidad=float(rng.uniform(0.28, 0.95)),
                viento=float(rng.uniform(-0.5, 0.5)),
                semilla=int(rng.integers(1e6))))

    polaridad = 'oscuro' if rng.random() < 0.75 else 'claro'
    fuerza = float(rng.uniform(0.50, 0.78))

    frames, cajas = [], []
    for i in range(n_frames):
        t = i / fps
        # temblor de cámara en mano
        f = aplicar_temblor(fondo, rng.normal(0, 1.4), rng.normal(0, 1.1),
                            rot=rng.normal(0, 0.25))
        f = perturbar(f, rng)

        cajas_frame = []
        for p in penachos:
            m = p.mascara(t)
            f = componer(f, m, polaridad, fuerza)
            b = caja(m, umbral=0.12)
            if b and (b[2] - b[0]) > 6 and (b[3] - b[1]) > 8:
                cajas_frame.append(b)

        frames.append(f)
        cajas.append(cajas_frame)

    return frames, cajas


def a_yolo(b, ancho, alto):
    x1, y1, x2, y2 = b
    return (((x1 + x2) / 2) / ancho, ((y1 + y2) / 2) / alto,
            (x2 - x1) / ancho, (y2 - y1) / alto)


def construir(destino='dataset', n_secuencias=60, prop_negativas=0.3,
              alto=256, ancho=448, n_frames=10, semilla=0):
    """Arma el conjunto completo, repartido en entrenamiento y validación."""
    rng = np.random.default_rng(semilla)
    for sub in ('train', 'val'):
        os.makedirs(f'{destino}/images/{sub}', exist_ok=True)
        os.makedirs(f'{destino}/labels/{sub}', exist_ok=True)

    stats = dict(frames=0, con_fuga=0, sin_fuga=0, cajas=0)

    for s in range(n_secuencias):
        con_fuga = rng.random() > prop_negativas
        sub = 'val' if s % 5 == 0 else 'train'          # 20% a validación
        frames, cajas = generar_secuencia(
            alto, ancho, n_frames, con_fuga, semilla=int(rng.integers(1e6)))

        for i, (f, cs) in enumerate(zip(frames, cajas)):
            nombre = f'seq{s:03d}_{i:02d}'
            cv2.imwrite(f'{destino}/images/{sub}/{nombre}.png', f)
            with open(f'{destino}/labels/{sub}/{nombre}.txt', 'w') as fh:
                for b in cs:
                    cx, cy, w, h = a_yolo(b, ancho, alto)
                    fh.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
            stats['frames'] += 1
            stats['cajas'] += len(cs)
        stats['con_fuga' if con_fuga else 'sin_fuga'] += 1

    with open(f'{destino}/data.yaml', 'w') as fh:
        fh.write(f'path: {os.path.abspath(destino)}\n'
                 'train: images/train\nval: images/val\n\n'
                 'nc: 1\nnames: [fuga_gas]\n')

    with open(f'{destino}/stats.json', 'w') as fh:
        json.dump(stats, fh, indent=1)

    return stats


if __name__ == '__main__':
    print(construir(n_secuencias=40))
