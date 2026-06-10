import os
import time
import math
import numpy as np
from PIL import Image

from src.scene import Sphere, Box, Torus, Light, Camera, Scene
from src.raytracer import normalize
from src.montecarlo import estimate_pixel_color

# --- Parâmetros de renderização ---
WIDTH  = 400
HEIGHT = 400
N      = 100  # amostras por pixel: 1=granulado · 10=razoável · 100=suave · 500+=quase sem ruído
SEED   = 42

# --- Opções do menu interativo ---
SHAPES = {
    "1": "Esfera",
    "2": "Cubo",
    "3": "Rosquinha",
}

COLORS = {
    "1": ("Vermelho", np.array([0.85, 0.10, 0.10])),
    "2": ("Azul",     np.array([0.10, 0.25, 0.85])),
    "3": ("Verde",    np.array([0.10, 0.75, 0.20])),
    "4": ("Amarelo",  np.array([0.90, 0.80, 0.10])),
}

LIGHTS = {
    "1": ("Esquerda-alta",  np.array([-3.0,  5.0, -1.0])),
    "2": ("Direita-alta",   np.array([ 3.0,  5.0, -1.0])),
    "3": ("Frontal-baixo",  np.array([ 0.0, -2.0,  2.0])),
    "4": ("Traseira-alta",  np.array([ 0.0,  5.0,  2.0])),
}


def _choose(prompt, options):
    """Exibe um menu numerado e retorna a chave escolhida (valida até acertar)."""
    keys = list(options.keys())
    while True:
        print(prompt)
        for k, v in options.items():
            label = v if isinstance(v, str) else v[0]
            print(f"  {k}. {label}")
        choice = input(f"Escolha [{keys[0]}-{keys[-1]}]: ").strip()
        if choice in options:
            return choice
        print(f"  Opção inválida. Tente novamente.\n")


def ask_config():
    """Pergunta ao usuário as opções da cena e retorna um dicionário de configuração."""
    print("\n=== Configuração da cena ===\n")

    shape_key    = _choose("Forma do objeto:", SHAPES)
    color_key    = _choose("\nCor do objeto:", COLORS)
    light_key    = _choose("\nPosição da luz:", LIGHTS)
    filename     = input("\nNome do arquivo de saída (Enter para \"output\"): ").strip()

    if not filename:
        filename = "output"

    return {
        "shape":    shape_key,
        "color":    COLORS[color_key][1],
        "light":    LIGHTS[light_key][1],
        "filename": filename,
    }


def build_scene(config):
    color = config["color"]

    if config["shape"] == "1":   # Esfera — câmera frontal padrão
        shape = Sphere(
            center=np.array([0.0, 0.0, -3.0]),
            radius=1.0,
            color=color,
        )
        camera = Camera(
            position=np.array([0.0, 0.0,  0.0]),
            look_at =np.array([0.0, 0.0, -1.0]),
            fov=60.0,
        )

    elif config["shape"] == "2":  # Cubo — câmera isométrica (vê topo + direita + frente)
        shape = Box(
            min_corner=np.array([-0.8, -0.8, -3.8]),
            max_corner=np.array([ 0.8,  0.8, -2.2]),
            color=color,
        )
        camera = Camera(
            position=np.array([1.9, 1.9, -1.1]),
            look_at =np.array([0.0, 0.0, -3.0]),
            fov=60.0,
        )

    else:                         # Rosquinha — câmera de cima-frente (vê o anel claramente)
        shape = Torus(
            center=np.array([0.0, 0.0, -3.0]),
            R=0.70,   # raio maior
            r=0.28,   # raio do tubo
            color=color,
        )
        camera = Camera(
            position=np.array([0.0, 2.5,  0.5]),
            look_at =np.array([0.0, 0.0, -3.0]),
            fov=60.0,
        )

    light = Light(position=config["light"], intensity=1.0)
    return Scene(shape, light, camera)


def camera_basis(camera):
    """Computa os eixos ortogonais da câmera (forward, right, up)."""
    forward = normalize(camera.look_at - camera.position)
    right   = normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    up      = np.cross(right, forward)  # já normalizado pois right ⊥ forward
    return forward, right, up


def main():
    config = ask_config()
    output = f"data/{config['filename']}.png"

    scene = build_scene(config)
    rng   = np.random.default_rng(SEED)

    forward, cam_right, cam_up = camera_basis(scene.camera)

    # Dimensões do plano da imagem (câmera pinhole; plano projetivo a distância 1)
    half_h     = math.tan(math.radians(scene.camera.fov / 2))
    half_w     = half_h * (WIDTH / HEIGHT)
    pixel_size = 2 * half_h / HEIGHT  # tamanho de um pixel em coordenadas de mundo

    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    print(f"\nRenderizando {WIDTH}x{HEIGHT} com N={N} amostras por pixel...")
    t0 = time.perf_counter()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Coordenadas normalizadas do pixel: u ∈ [-1,1], v ∈ [1,-1] (y invertido)
            u = (x + 0.5) / WIDTH  * 2 - 1
            v = 1 - (y + 0.5) / HEIGHT * 2

            base_dir = forward + u * half_w * cam_right + v * half_h * cam_up

            pixels[y, x] = estimate_pixel_color(
                scene.camera.position, base_dir,
                cam_right, cam_up, pixel_size,
                scene, N, rng,
            )

    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(output), exist_ok=True)
    img = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8), mode="RGB")
    img.save(output)

    print(f"Tempo total: {elapsed:.1f}s")
    print(f"Imagem salva em {output}")


if __name__ == "__main__":
    main()
