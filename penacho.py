"""
Generador de penachos de gas sintéticos para video OGI
======================================================
Produce secuencias de un penacho ascendente con la apariencia que tiene
el metano en una cámara de imagen óptica de gas: una nube difusa, sin
bordes definidos, que sube, se abre y se deforma con la turbulencia.

El objetivo es entrenar un detector sin tener que etiquetar video real
a mano: aquí la máscara del penacho se conoce con exactitud, porque la
generamos nosotros.
"""

import numpy as np


# ---------------------------------------------------------------- ruido
def _interp_suave(t):
    """Curva de Perlin: suaviza la interpolación entre nodos de la rejilla."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def ruido_perlin(alto, ancho, escala=8, semilla=None):
    """
    Ruido de Perlin en 2D, normalizado a [0, 1].
    Es la base de la turbulencia: da manchas orgánicas en vez de estática.
    """
    rng = np.random.default_rng(semilla)
    gy, gx = escala + 1, escala + 1

    # gradientes unitarios aleatorios en cada nodo
    ang = rng.uniform(0, 2 * np.pi, (gy, gx))
    grad = np.dstack((np.cos(ang), np.sin(ang)))

    ys = np.linspace(0, escala, alto, endpoint=False)
    xs = np.linspace(0, escala, ancho, endpoint=False)
    xx, yy = np.meshgrid(xs, ys)

    x0, y0 = xx.astype(int), yy.astype(int)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = xx - x0, yy - y0

    def punto(gxi, gyi, ddx, ddy):
        g = grad[gyi, gxi]
        return g[..., 0] * ddx + g[..., 1] * ddy

    n00 = punto(x0, y0, dx,     dy)
    n10 = punto(x1, y0, dx - 1, dy)
    n01 = punto(x0, y1, dx,     dy - 1)
    n11 = punto(x1, y1, dx - 1, dy - 1)

    u, v = _interp_suave(dx), _interp_suave(dy)
    arriba = n00 * (1 - u) + n10 * u
    abajo  = n01 * (1 - u) + n11 * u
    val = arriba * (1 - v) + abajo * v

    return (val - val.min()) / (np.ptp(val) + 1e-9)


def ruido_fractal(alto, ancho, octavas=4, escala=4, semilla=None):
    """
    Suma de varias capas de Perlin a distinta escala.
    Las capas grandes dan la forma general; las pequeñas, el detalle fino.
    """
    total = np.zeros((alto, ancho), np.float32)
    amplitud, norma = 1.0, 0.0
    for o in range(octavas):
        s = escala * (2 ** o)
        sem = None if semilla is None else semilla + o * 977
        total += amplitud * ruido_perlin(alto, ancho, escala=s, semilla=sem)
        norma += amplitud
        amplitud *= 0.5
    return total / norma


# ---------------------------------------------------------------- penacho
class Penacho:
    """
    Un penacho de gas visto por la cámara OGI.

    intensidad : 0 a 1. Aproxima el caudal de la fuga: una fuga fuerte
                 sube más rápido, se ve más densa y llega más alto.
    viento     : desplazamiento horizontal por unidad de altura.
    """

    def __init__(self, alto, ancho, origen, intensidad=0.6,
                 viento=0.35, semilla=None):
        self.alto, self.ancho = alto, ancho
        self.ox, self.oy = origen
        self.intensidad = float(np.clip(intensidad, 0.05, 1.0))
        self.viento = viento
        self.rng = np.random.default_rng(semilla)
        self.semilla = semilla if semilla is not None else self.rng.integers(1e6)

        # campo de turbulencia propio de este penacho
        self.turbulencia = ruido_fractal(alto, ancho, octavas=4, escala=3,
                                         semilla=int(self.semilla))

        # geometría: a más caudal, penacho más alto y más ancho
        self.altura_max = (0.28 + 0.46 * self.intensidad) * alto
        self.ancho_base = (0.012 + 0.014 * self.intensidad) * ancho
        self.apertura   = (0.5 + 0.8 * self.intensidad)

    def mascara(self, t):
        """
        Devuelve la máscara del penacho en el instante t (segundos).
        Valores 0 a 1: cuánto gas hay en cada píxel.
        """
        ys = np.arange(self.alto)[:, None]
        xs = np.arange(self.ancho)[None, :]

        # altura relativa sobre el punto de fuga (1 en la boquilla, 0 arriba)
        h = (self.oy - ys) / max(self.altura_max, 1)
        dentro = (h >= 0) & (h <= 1)

        # el penacho se abre en cono a medida que sube
        semiancho = self.ancho_base * (1 + self.apertura * h * 2.4)

        # serpenteo: el eje no sube recto, oscila con el tiempo
        fase = t * (0.7 + 0.9 * self.intensidad)
        vaiven = (np.sin(h * 5.5 + fase * 2.2) * 0.35 +
                  np.sin(h * 2.3 - fase * 1.4) * 0.22)
        eje = self.ox + self.viento * h * self.altura_max + vaiven * semiancho * 2.2

        # perfil gaussiano alrededor del eje
        d = (xs - eje) / np.maximum(semiancho, 1e-6)
        m = np.exp(-0.5 * d ** 2)

        # se desvanece con la altura: el gas se diluye al alejarse
        m *= np.clip(1 - h, 0, 1) ** 0.85
        m *= dentro

        # turbulencia: la nube no es lisa, se rompe en grumos que ascienden
        desp = int((t * (26 + 42 * self.intensidad)) % self.alto)
        turb = np.roll(self.turbulencia, desp, axis=0)
        m *= (0.55 + 0.8 * turb)

        # arranque más denso junto a la boquilla
        m += np.exp(-0.5 * (d * 1.7) ** 2) * np.clip(1 - h * 4.5, 0, 1) * 0.55 * dentro

        return np.clip(m * (0.35 + 0.65 * self.intensidad), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- composición
def componer(fondo, mascara, polaridad='oscuro', fuerza=0.55):
    """
    Superpone el penacho sobre un fotograma OGI en escala de grises.

    polaridad: en una cámara OGI el gas puede verse más oscuro o más claro
               que el fondo, según el modo y la temperatura de la escena.
               El detector debe aprender ambos casos.
    """
    f = fondo.astype(np.float32)
    m = mascara * fuerza

    if polaridad == 'oscuro':
        salida = f * (1 - m) + (f * 0.35) * m
    else:
        salida = f * (1 - m) + (f + (255 - f) * 0.75) * m

    # el gas difumina los bordes del fondo: se pierde nitidez donde hay nube
    import cv2
    borroso = cv2.GaussianBlur(f, (0, 0), 2.2)
    peso = np.clip(m * 1.6, 0, 1)
    salida = salida * (1 - peso * 0.45) + borroso * (peso * 0.45)

    return np.clip(salida, 0, 255).astype(np.uint8)


def caja(mascara, umbral=0.10):
    """Rectángulo que encierra el penacho, para la etiqueta de entrenamiento."""
    ys, xs = np.where(mascara > umbral)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
