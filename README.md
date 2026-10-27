<h1 align="center">KitchGen</h1>

<p align="center">
  <b>Text → a 3D kitchen object → the point cloud a stereo depth camera would see.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/objects-12-1baf7a" alt="12 objects">
</p>

<table align="center">
  <tr>
    <td align="center"><img src="glass2.png" height="200"><br><sub>mesh</sub></td>
    <td align="center"><img src="glass.png" height="200"><br><sub>what the camera saw (orange line = camera)</sub></td>
    <td align="center"><img src="glass1.png" height="200"><br><sub>both together</sub></td>
  </tr>
</table>

<p align="center"><code>python generate.py "twisted square glass" --neural</code></p>

---

KitchGen turns a plain English description into a mesh of a kitchen object, then photographs it with
a simulated stereo depth camera. You get the clean shape and the noisy, partial point cloud a real sensor would
give you. That pair is handy for training and testing 3D perception models without a single real photo.

- **12 objects:** fork, spoon, knife, ladle, spatula, mug, cup (optionally on a saucer), bowl, plate, glass, wine glass, frying pan
- **Any outline:** square, oval, hexagonal, fluted, twisted, barrel-shaped containers
- **Crosses:** a mug-bowl hybrid, "70% mug 30% wine glass", a spork
- **A realistic camera:** modelled on an Intel RealSense D435, with stereo shadows, depth noise, holes, flying pixels and material effects (metal glare, see-through glass)
- **Two shape engines:** exact formulas, or a trained neural network that has learned all 12 objects

## Quick start

```
git clone https://github.com/Sulimowski/Kitch_Gen.git
cd Kitch_Gen
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python generate.py "fork with 3 prongs and a long handle"
```

A browser tab opens with the mesh, the point cloud and the camera (drag to rotate; click a legend entry to hide
or show a layer). The trained network is included in `models/`, so `--neural` works straight away.

Every run saves its results to `outputs/<object>_<time>/`:

| file | what it is |
|---|---|
| `mesh.obj` | the object's mesh |
| `cloud.ply` | the stereo point cloud in object coordinates (lines up with the mesh) |
| `cloud_camera.ply` | the same cloud in camera coordinates, the way a real sensor hands it to you |
| `depth.png` | the depth frame |
| `params.json` | what was understood from the text, every shape parameter, the camera pose and intrinsics |
| `view.html` | the 3D viewer |

<p align="center">
  <img src="depth1.png" width="400">
  <img src="depth.png" width="400"><br>
  <sub>Stereo point clouds coloured by distance from the camera: only the side facing the camera is there.</sub>
</p>

## What you can ask for

```
python generate.py "teaspoon"
python generate.py "8 inch chef's knife"
python generate.py "slotted spatula with a thick handle"
python generate.py "350 ml mug"
python generate.py "teacup with a saucer"
python generate.py "27 cm dinner plate"
python generate.py "champagne flute"
python generate.py "12 inch frying pan"
```

The parser understands:

| kind | examples |
|---|---|
| objects and presets | teaspoon, chef's / bread / steak knife, cleaver, teacup, travel mug, beer stein, pint glass, champagne flute, martini glass, soup bowl, side plate, wok… |
| counts | "3 prongs", "three-pronged", "4-tine", "5 slots" |
| sizes | "20 cm tablespoon", "8 inch knife", "10 cm tall", "15 cm wide" |
| capacity | "350 ml mug", "12 oz coffee mug", "1 l bowl" |
| features | with / without a handle, on a saucer, with a hook, serrated, slotted |
| adjectives | long, wide, thick, deep, tall, sharp, curved, flared… applied to the right part: "long handle", "deep bowl", "long stem" |
| materials | glass, steel, ceramic, porcelain, plastic, wooden (they change what the camera sees) |

Anything you leave out is picked at random within realistic limits.

### Shapes

Containers (mug, cup, glass, wine glass, bowl, plate, frying pan) can take any outline:

```
python generate.py "square mug"
python generate.py "slightly oval mug with a big handle"
python generate.py "hexagonal glass"           # also triangular, pentagonal, octagonal, "6-sided"
python generate.py "fluted bowl"               # wavy outline; also "scalloped", "star", "flower"
python generate.py "twisted square glass"
python generate.py "barrel mug"                # bulging sides; "hourglass" pinches them in
python generate.py "square cup on a saucer"    # the saucer follows the cup's outline
```

"slightly" and "very" set the strength: from a rounded "slightly square" to a hard "very square".

### Crosses

```
python generate.py "cross between a mug and a bowl"
python generate.py "70% mug 30% wine glass"
python generate.py "mostly a glass, a bit of a bowl"
python generate.py "spork"                     # also "knork" (knife + fork), "splayd" (spoon + knife)
```

Containers cross with containers, and fork, spoon, knife and spatula cross with each other.

### Options

| flag | effect |
|---|---|
| `--neural` | the trained neural network builds the shape instead of the formulas |
| `--table` | the object stands on a table, and the table is in the point cloud too |
| `--views N` | N cameras shoot the object from different sides; all views are fused into `cloud.ply` |
| `--count N` | generate N different objects matching the description (a small dataset) |
| `--wild` | containers also get a random outline unless the text sets one |
| `--seed N` | the same seed gives the same result |
| `--no-view` | do not open the browser |
| `--out DIR` | output folder (default `outputs`) |

## How it works

```mermaid
flowchart LR
    T(["text"]) --> P["parser<br/>text.py"]
    P --> S["SDF from formulas<br/>shapes.py, hybrid.py"]
    P --> N["SDF from the network<br/>neural.py"]
    S --> M["marching cubes<br/>mesh.py"]
    N --> M
    M --> C["stereo camera<br/>stereo.py"]
    C --> V(["files + viewer<br/>viz.py"])
```

**Shapes are signed distance functions (SDFs).** For any point in space they give the distance to the surface,
negative inside. Cutlery is built from flat 2D outlines extruded into plates and joined with smooth unions.
Containers are profiles revolved around the vertical axis, with ring handles and foot rings. Their "radius" is
not plain round distance:
- a superellipse norm makes it square;
- scaling one axis makes it oval;
- a polygon norm gives N sides;
- a cosine term makes it wavy;
- rotating it with height twists it.

**Crosses morph shapes instead of averaging distances.** Averaging two thin walls that don't overlap would
leave nothing. So container profiles are resampled by arc length and blended point by point, while cutlery
blends its 2D outline, thickness and curve.

**Meshes** come from marching cubes on a 0.6 to 1.2 mm grid.

**The stereo camera** works at 1280×720 with a 5 cm baseline. It sees only its own side of the object, and a
point must be visible to both lenses. Depth noise grows with distance squared, and disparity is rounded to 1/8
pixel. It also adds flying pixels on edges and random holes. Materials matter:
- metal blinds the camera where it faces it head-on;
- ceramic and plastic return clean data;
- glass is mostly invisible, so the camera often sees the table straight through it.

## The neural network

The network is a **conditional SDF** of about 500k parameters.
- **Input:** a point (x, y, z), its *warp radius* (the distance from the container's axis measured in the
  object's own square, oval or twisted outline) and a description of the object (kind and parameters).
- **Output:** the distance from that point to the surface.

<p align="center">
  <img src="net_square_mug.png" width="760"><br>
  <sub><code>python show_network.py "square mug"</code>: every neuron coloured by what it outputs for one point on the mug's wall</sub>
</p>

- **Top row (the main path):** point → 84 Fourier features → 6 layers of 256 neurons → distance.
  The Fourier features let it draw thin rims and prongs, and a skip connection in the middle feeds the
  features in again.
- **Bottom row:** the object's description squeezed into a 64-number code that is fed into every hidden layer.
  The same network draws a fork, a wine glass or a square mug.

It is trained on 6000 procedurally generated objects × 10 000 points. Its mean error on objects it has never
seen is about 0.13 mm.

<p align="center">
  <img src="net950.png" width="760"><br>
  <sub>The same network from the side: lines are the strongest weights, blue positive, red negative</sub>
</p>

### Train it yourself

```
python train.py                   # build the training data (once, ~7 min), then train (~50 min on an RTX 3060)
python train.py --steps 120000    # train longer for crisper prongs and rims
python train.py --rebuild         # rebuild the training data
```

Training saves a checkpoint every 2000 steps and resumes from it if it was interrupted. When it is done,
`outputs/neural_check.png` shows each object built by the formulas next to the network's version.

To look inside the network:

```
python show_network.py "fork"                   # a window you can rotate with the mouse
python show_network.py "oval mug" --save net.png
```

## Project layout

```
generate.py        text → mesh + point cloud (the main CLI)
train.py           build the dataset and train the network
show_network.py    3D picture of the trained network
kitchgen/
  text.py          English prompt parser
  shapes.py        procedural SDFs of all 12 objects and their outlines
  hybrid.py        crosses between objects
  mesh.py          marching cubes, .obj / .ply export
  stereo.py        stereo depth camera simulator
  condition.py     how an object is described to the network
  dataset.py       training data
  neural.py        the conditional SDF network
  viz.py           browser viewer, depth image
netviz/            3D neural network drawing
models/            the trained network
```
