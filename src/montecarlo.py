import numpy as np
from .raytracer import normalize, trace_rays


def estimate_pixel_color(ray_origin, base_dir, cam_right, cam_up, pixel_size, scene, N, rng):
    """
    Estima a cor de um pixel disparando N raios com perturbações aleatórias.

    Integração por Monte Carlo:
        cor_pixel ≈ (1/N) Σ f(raio_i)
    onde raio_i são amostras uniformes dentro da área do pixel.
    """
    # 1. AMOSTRAGEM: N offsets aleatórios no plano da imagem (jitter uniforme em [-0.5, 0.5] pixels)
    jitter = (rng.random((N, 2)) - 0.5) * pixel_size  # (N, 2) em unidades de mundo

    # Aplica o jitter usando os eixos da câmera para manter o deslocamento no plano da imagem
    dirs = (base_dir
            + jitter[:, 0:1] * cam_right   # deslocamento horizontal
            + jitter[:, 1:2] * cam_up)     # deslocamento vertical
    dirs = normalize(dirs)  # (N, 3)

    # 2. Avalia f em cada amostra: traça os N raios pela cena
    colors = trace_rays(ray_origin, dirs, scene)  # (N, 3)

    # 3. ESTIMADOR DE MONTE CARLO: média = (1/N) Σ f(raio_i)
    return colors.mean(axis=0)  # (3,)
