# raytracing-montecarlo
Ray tracing com integração numérica por Monte Carlo. Cada pixel é calculado disparando **N raios aleatórios** e calculando a média das cores — isso é a rendering equation resolvida via Monte Carlo: `cor ≈ (1/N) Σ f(raio_i)`.

## Como rodar

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Estrutura

| Arquivo | Responsabilidade |
|---|---|
| `scene.py` | Estruturas de dados (Sphere, Light, Camera, Scene) |
| `raytracer.py` | Física: interseção, normais, shading difuso — a função `f(raio)` |
| `montecarlo.py` | Integração: amostragem, avaliação e média — o estimador `(1/N) Σ f` |
| `main.py` | Orquestração: monta a cena, itera pixels, salva `output.png` |

## Convergência do Monte Carlo

Altere `N` no topo de `main.py` para ver o efeito visualmente:

| N | Resultado |
|---|---|
| 1 | Granulado (ruído visível) |
| 10 | Melhor, ainda com ruído |
| 100 | Suave, resultado limpo |
| 500+ | Quase sem ruído |
