import numpy as np
from .scene import Sphere, Box, Torus


def normalize(v):
    """Normaliza vetor(es) ao longo do último eixo."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Interseção — Esfera
# ---------------------------------------------------------------------------

def ray_sphere_intersect(origin, dirs, sphere):
    """
    Retorna t (N,): distância de interseção para cada raio.
    np.inf significa que o raio não acertou a esfera.

    Resolve: |origin + t*dir - center|^2 = r^2
    Com dirs normalizados (a=1): t^2 + 2*(dir·oc)*t + (|oc|^2 - r^2) = 0
    """
    oc = origin - sphere.center               # (3,)
    b  = 2.0 * (dirs @ oc)                    # (N,)
    c  = np.dot(oc, oc) - sphere.radius ** 2  # escalar
    discriminant = b ** 2 - 4.0 * c           # (N,)

    t = np.full(len(dirs), np.inf)
    hit = discriminant >= 0
    if not np.any(hit):
        return t

    sqrt_d = np.sqrt(discriminant[hit])
    t_near = (-b[hit] - sqrt_d) / 2.0
    t_far  = (-b[hit] + sqrt_d) / 2.0

    # prefere a raiz menor; cai para a maior se t_near <= 0 (câmera dentro da esfera)
    t_candidate = np.where(t_near > 1e-6, t_near, t_far)
    t[hit] = np.where(t_candidate > 1e-6, t_candidate, np.inf)
    return t


def compute_sphere_normals(hit_points, sphere):
    """Normal de superfície (apontando para fora) em cada ponto de hit."""
    return normalize(hit_points - sphere.center)


# ---------------------------------------------------------------------------
# Interseção — Cubo (AABB, método das slabs)
# ---------------------------------------------------------------------------

def ray_box_intersect(origin, dirs, box):
    """
    Retorna (t, normals) para N raios contra um AABB.
    t[i] = np.inf se o raio i não acertou.
    normals[i] = normal da face de entrada para os raios que acertaram.

    Método das slabs: para cada eixo, calcula o intervalo [t_near, t_far].
    O raio acerta a caixa se max(t_near) < min(t_far) e a entrada é positiva.
    """
    inv_dirs = np.where(np.abs(dirs) > 1e-12, 1.0 / dirs, np.inf)  # (N, 3)

    t0 = (box.min_corner - origin) * inv_dirs  # (N, 3)
    t1 = (box.max_corner - origin) * inv_dirs  # (N, 3)

    t_near = np.minimum(t0, t1)  # (N, 3) — entrada por eixo
    t_far  = np.maximum(t0, t1)  # (N, 3) — saída por eixo

    t_enter = t_near.max(axis=1)   # (N,) — entrada real no cubo
    t_exit  = t_far.min(axis=1)    # (N,) — saída real do cubo

    t = np.full(len(dirs), np.inf)
    hit = (t_exit >= t_enter) & (t_enter > 1e-6)
    t[hit] = t_enter[hit]

    # normal: eixo que determinou t_enter (face de entrada)
    normals = np.zeros((len(dirs), 3))
    entry_axis = np.argmax(t_near, axis=1)  # (N,) — eixo 0, 1 ou 2
    for ax in range(3):
        mask = hit & (entry_axis == ax)
        if np.any(mask):
            normals[mask, ax] = -np.sign(dirs[mask, ax])

    return t, normals


# ---------------------------------------------------------------------------
# Interseção — Toro (rosquinha)
# ---------------------------------------------------------------------------

def ray_torus_intersect(origin, dirs, torus):
    """
    Retorna t (N,) para N raios contra um toro de eixo Y.
    np.inf = miss.

    Substitui P = O' + t*D (O' = origin - torus.center) na equação implícita:
        (|P|² + R² - r²)² - 4·R²·(Px² + Pz²) = 0
    obtendo o quartico: t⁴ + B·t³ + C·t² + D·t + E = 0.

    As raízes são obtidas como autovalores da matriz companion (forma Frobenius),
    vetorizados sobre os N raios com np.linalg.eigvals.
    """
    O  = origin - torus.center   # translate para o espaço local do toro
    R, r = torus.R, torus.r

    # grandezas por raio (shape N)
    b = dirs @ O                                    # O·D
    p = dirs[:, 0] ** 2 + dirs[:, 2] ** 2          # Dx² + Dz²
    q = O[0] * dirs[:, 0] + O[2] * dirs[:, 2]      # Ox·Dx + Oz·Dz

    # grandezas constantes (mesma origem para todos os raios)
    c = float(np.dot(O, O)) + R**2 - r**2          # |O|² + R² - r²
    s = float(O[0]**2 + O[2]**2)                   # Ox² + Oz²

    # coeficientes do quartico t⁴ + coef3·t³ + coef2·t² + coef1·t + coef0 = 0
    coef3 =  4 * b
    coef2 =  4 * b**2 + 2*c - 4*R**2 * p
    coef1 =  4 * b*c  - 8*R**2 * q
    coef0 =  c**2     - 4*R**2 * s     # escalar

    # matriz companion de Frobenius (N × 4 × 4)
    # autovalores = raízes do quartico
    N = len(dirs)
    comp = np.zeros((N, 4, 4))
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    comp[:, 3, 2] = 1.0
    comp[:, 0, 3] = -coef0    # escalar → broadcast
    comp[:, 1, 3] = -coef1    # (N,)
    comp[:, 2, 3] = -coef2    # (N,)
    comp[:, 3, 3] = -coef3    # (N,)

    roots = np.linalg.eigvals(comp)   # (N, 4) complexo

    # filtra raízes reais positivas (raízes verdadeiramente complexas têm |imag| grande)
    real_mask = np.abs(roots.imag) < 1e-4
    real_t    = np.where(real_mask, roots.real, np.inf)
    pos_t     = np.where(real_t > 1e-6, real_t, np.inf)
    return pos_t.min(axis=1)   # (N,) menor t positivo real por raio


def compute_torus_normals(hit_points, torus):
    """
    Gradiente da função implícita do toro (eixo Y), normalizado.
        ∇F = 4·(|P|²+R²−r²)·P − 8·R²·(Px, 0, Pz)
    """
    P  = hit_points - torus.center                          # (M, 3)
    c  = np.sum(P**2, axis=1, keepdims=True) + torus.R**2 - torus.r**2
    Pxz = P * np.array([1.0, 0.0, 1.0])                    # zera componente Y
    return normalize(4 * c * P - 8 * torus.R**2 * Pxz)


# ---------------------------------------------------------------------------
# Shading e pipeline principal
# ---------------------------------------------------------------------------

def shade(hit_points, normals, shape, light):
    """
    Iluminação difusa de Lambert:
        cor = cor_objeto * max(N·L, 0) * intensidade
    """
    to_light  = normalize(light.position - hit_points)             # (M, 3)
    cos_theta = np.sum(normals * to_light, axis=1, keepdims=True)  # (M, 1)
    cos_theta = np.maximum(cos_theta, 0.0)
    return shape.color * cos_theta * light.intensity               # (M, 3)


def trace_rays(origin, dirs, scene):
    """
    Avalia f(raio) para N raios. Retorna cores (N, 3).
    Esta é a função 'f' que o Monte Carlo vai integrar.
    """
    shape    = scene.shape
    colors   = np.zeros((len(dirs), 3))
    box_normals = None

    if isinstance(shape, Sphere):
        t = ray_sphere_intersect(origin, dirs, shape)
    elif isinstance(shape, Box):
        t, box_normals = ray_box_intersect(origin, dirs, shape)
    elif isinstance(shape, Torus):
        t = ray_torus_intersect(origin, dirs, shape)
    else:
        return colors

    hit_mask = np.isfinite(t)
    if not np.any(hit_mask):
        return colors

    hit_points = origin + t[hit_mask, np.newaxis] * dirs[hit_mask]  # (M, 3)

    if isinstance(shape, Sphere):
        normals = compute_sphere_normals(hit_points, shape)
    elif isinstance(shape, Box):
        normals = box_normals[hit_mask]
    elif isinstance(shape, Torus):
        normals = compute_torus_normals(hit_points, shape)

    colors[hit_mask] = shade(hit_points, normals, shape, scene.light)
    return colors
