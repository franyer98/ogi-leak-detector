"""
Fondos sintéticos con el aspecto de una escena OGI real.

Una cámara térmica no ve formas limpias: ve superficies a distinta
temperatura, tuberías cruzando el cuadro, vegetación caliente al sol,
viñeteado del lente y ruido de sensor. Todo eso importa, porque el
detector debe aprender a distinguir el gas de ese desorden y no de un
fondo plano.
"""

import numpy as np
import cv2
from penacho import ruido_fractal


def _rect(img, x1, y1, x2, y2, val):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), float(val), -1)


def _tanque(img, rng, x, ancho_t, y_techo, y_suelo, temp):
    """Tanque vertical con anillos, accesorios y estructura."""
    alto_t = y_suelo - y_techo

    # cuerpo con gradiente lateral: un costado recibe más sol
    for i in range(int(ancho_t)):
        f = i / max(ancho_t, 1)
        lat = np.sin(f * np.pi) ** 0.6
        _rect(img, x + i, y_techo, x + i + 1, y_suelo, temp * (0.82 + 0.28 * lat))

    cv2.ellipse(img, (int(x + ancho_t / 2), int(y_techo)),
                (int(ancho_t / 2), int(alto_t * 0.09)),
                0, 180, 360, float(temp * 1.10), -1)

    # anillos entre virolas
    for f in (0.26, 0.48, 0.70, 0.89):
        y = y_techo + alto_t * f
        cv2.line(img, (int(x), int(y)), (int(x + ancho_t), int(y)),
                 float(temp * 0.86), max(1, int(alto_t * 0.012)))

    # manchas térmicas: nivel de producto, humedad, óxido
    for _ in range(rng.integers(3, 7)):
        mx = x + rng.uniform(0.1, 0.9) * ancho_t
        my = y_techo + rng.uniform(0.15, 0.95) * alto_t
        r = rng.uniform(0.06, 0.20) * ancho_t
        capa = np.zeros_like(img)
        cv2.circle(capa, (int(mx), int(my)), int(r), float(rng.uniform(-18, 16)), -1)
        img += cv2.GaussianBlur(capa, (0, 0), max(r * 0.55, 0.6))

    # accesorios del techo: candidatos a fuga
    accesorios = []
    n = int(rng.integers(2, 5))
    for k in range(n):
        ax = x + ancho_t * (0.16 + 0.68 * (k + 0.5) / n) + rng.uniform(-4, 4)
        ay = y_techo - alto_t * 0.028
        rr = ancho_t * rng.uniform(0.045, 0.075)
        cv2.ellipse(img, (int(ax), int(ay)), (int(rr), int(rr * 0.42)),
                    0, 0, 360, float(temp * rng.uniform(1.10, 1.26)), -1)
        _rect(img, ax - rr * 0.5, ay, ax + rr * 0.5, ay + alto_t * 0.03, temp * 1.15)
        accesorios.append((int(ax), int(ay)))

    # barandilla del techo
    yb = y_techo - alto_t * 0.085
    cv2.line(img, (int(x + ancho_t * 0.06), int(yb)),
             (int(x + ancho_t * 0.94), int(yb)), float(temp * 1.20), 1)
    for k in range(6):
        px = x + ancho_t * (0.08 + 0.84 * k / 5)
        cv2.line(img, (int(px), int(yb)), (int(px), int(y_techo)), float(temp * 1.18), 1)

    # escalera lateral
    ex = x + ancho_t + ancho_t * 0.03
    cv2.line(img, (int(ex), int(y_suelo)), (int(ex), int(y_techo)), float(temp * 1.22), 2)
    for k in range(10):
        y = y_techo + alto_t * k / 10
        cv2.line(img, (int(ex - ancho_t * 0.035), int(y)),
                 (int(ex + ancho_t * 0.035), int(y)), float(temp * 1.18), 1)

    return accesorios


def escena_ogi(alto=256, ancho=448, semilla=None):
    """Escena industrial en infrarrojo. Devuelve (imagen, accesorios)."""
    rng = np.random.default_rng(semilla)
    img = np.zeros((alto, ancho), np.float32)

    # --- cielo ---
    horizonte = int(alto * rng.uniform(0.66, 0.78))
    cielo = np.linspace(rng.uniform(52, 74), rng.uniform(92, 112), horizonte)
    img[:horizonte, :] = cielo[:, None]
    img[:horizonte, :] += (ruido_fractal(horizonte, ancho, octavas=3, escala=2,
                                         semilla=None if semilla is None else semilla + 5) - 0.5) * 22

    # --- terreno ---
    img[horizonte:, :] = rng.uniform(120, 148)
    img[horizonte:, :] += (ruido_fractal(alto - horizonte, ancho, octavas=4, escala=5,
                                         semilla=None if semilla is None else semilla + 9) - 0.5) * 30

    # vegetación: manchas cálidas irregulares
    for _ in range(int(rng.integers(5, 12))):
        vx = rng.uniform(0, ancho)
        vy = horizonte + rng.uniform(-3, (alto - horizonte) * 0.5)
        r = rng.uniform(8, 30)
        capa = np.zeros_like(img)
        cv2.circle(capa, (int(vx), int(vy)), int(r), float(rng.uniform(6, 22)), -1)
        img += cv2.GaussianBlur(capa, (0, 0), r * 0.6)

    # --- tanques ---
    accesorios = []
    n_tanques = int(rng.integers(1, 3))
    for k in range(n_tanques):
        ancho_t = ancho * rng.uniform(0.17, 0.30)
        x = ancho * (0.08 + 0.55 * k / max(n_tanques, 1)) + rng.uniform(-10, 10)
        x = float(np.clip(x, 4, ancho - ancho_t - 4))
        y_techo = horizonte - alto * rng.uniform(0.30, 0.46)
        accesorios += _tanque(img, rng, x, ancho_t, y_techo, horizonte,
                              rng.uniform(132, 168))

    # --- tubería horizontal con bridas ---
    if rng.random() < 0.7:
        yt = horizonte - alto * rng.uniform(0.04, 0.16)
        gr = rng.uniform(150, 182)
        _rect(img, 0, yt, ancho, yt + alto * 0.022, gr)
        for _ in range(int(rng.integers(2, 5))):
            bx = rng.uniform(0, ancho)
            _rect(img, bx - 4, yt - alto * 0.012, bx + 4, yt + alto * 0.034, gr * 1.10)

    # --- estructura vertical ---
    if rng.random() < 0.55:
        px = rng.uniform(ancho * 0.05, ancho * 0.95)
        _rect(img, px - 2, horizonte - alto * rng.uniform(0.35, 0.60), px + 2,
              horizonte, rng.uniform(150, 175))

    # --- efectos de cámara ---
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.2))

    # viñeteado del lente
    yy, xx = np.mgrid[0:alto, 0:ancho]
    r = np.sqrt(((yy - alto / 2) / (alto / 2)) ** 2 + ((xx - ancho / 2) / (ancho / 2)) ** 2)
    img *= (1 - 0.22 * np.clip(r, 0, 1.4) ** 2)

    # no uniformidad del sensor + grano
    img += rng.normal(0, 1.6, (alto, 1))
    img += rng.normal(0, rng.uniform(2.0, 4.2), (alto, ancho))

    return np.clip(img, 0, 255).astype(np.uint8), accesorios


def aplicar_temblor(img, dx, dy, rot=0.0):
    """Micro-movimiento de cámara en mano."""
    alto, ancho = img.shape
    M = cv2.getRotationMatrix2D((ancho / 2, alto / 2), rot, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (ancho, alto), borderMode=cv2.BORDER_REFLECT)


def perturbar(img, rng):
    """La ganancia automática de la cámara cambia el contraste entre cuadros."""
    f = img.astype(np.float32)
    f = (f - 128) * rng.uniform(0.92, 1.10) + 128 + rng.uniform(-6, 6)
    return np.clip(f, 0, 255).astype(np.uint8)
