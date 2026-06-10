from dataclasses import dataclass
import numpy as np


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray  # RGB em [0, 1]


@dataclass
class Box:
    min_corner: np.ndarray
    max_corner: np.ndarray
    color: np.ndarray  # RGB em [0, 1]


@dataclass
class Torus:
    center: np.ndarray
    R: float             # raio maior (do centro do toro ao centro do tubo)
    r: float             # raio menor (raio do tubo)
    color: np.ndarray    # RGB em [0, 1]


@dataclass
class Light:
    position: np.ndarray
    intensity: float


@dataclass
class Camera:
    position: np.ndarray
    look_at: np.ndarray  # ponto para onde a câmera aponta
    fov: float           # campo de visão vertical em graus


@dataclass
class Scene:
    shape: object   # Sphere | Box | Torus
    light: Light
    camera: Camera
