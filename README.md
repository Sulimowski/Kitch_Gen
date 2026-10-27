# KitchGen

Text → a 3D mesh of a kitchen object + a point cloud that looks like it was captured by a stereo depth camera.

```
python generate.py "fork with 3 prongs and a long handle"
```

A browser tab opens with the mesh, the point cloud and the camera on a black background (drag to rotate,
click a legend entry to hide or show a layer). Files go to `outputs/<object>_<time>/`:

| file | what it is |
|---|---|
| `mesh.obj` | the object's mesh |
| `cloud.ply` | the stereo point cloud in object coordinates (lines up with the mesh) |
| `cloud_camera.ply` | the same cloud in camera coordinates — the way a real sensor hands it to you |
| `depth.png` | the depth frame, as a sensor would see it |
| `params.json` | what was understood from the text, every shape parameter, the camera pose and intrinsics |
| `view.html` | the 3D viewer |

## Objects

fork, spoon, knife, ladle, spatula, mug, cup (optionally on a saucer), bowl, plate, glass, wine glass, frying pan.

## Example prompts

```
python generate.py "teaspoon"
python generate.py "big soup spoon with a deep bowl"
python generate.py "8 inch chef's knife"
python generate.py "serrated bread knife"
python generate.py "soup ladle with a hook"
python generate.py "slotted spatula with a thick handle"
python generate.py "350 ml mug"
python generate.py "tall mug without a handle"
python generate.py "teacup with a saucer"
python generate.py "deep soup bowl"
python generate.py "27 cm dinner plate"
python generate.py "pint glass"
python generate.py "champagne flute"
python generate.py "12 inch frying pan"
python generate.py "glass bowl"
```

The parser understands:
- **objects and presets**: teaspoon, tablespoon, chef's/bread/steak/butter/paring knife, cleaver, dinner/dessert/serving
  fork, espresso cup, teacup, travel mug, beer stein, shot/pint/highball/rocks glass, red/white wine glass, champagne
  flute, martini glass, soup/rice/salad bowl, dinner/soup/side plate, saucer, platter, wok, crepe pan…
- **counts**: "3 prongs", "three-pronged", "4-tine", "5 slots"
- **sizes**: cm, mm, inches ("20 cm tablespoon", "8 inch knife", "27 cm plate"), "10 cm tall", "15 cm wide"
- **capacity**: ml, cl, l, oz ("350 ml mug", "12 oz coffee mug")
- **features**: with/without a handle, on a saucer, with/without a hook, serrated, slotted
- **adjectives and the part they describe**: long/short, wide/narrow, thick/thin, deep/shallow, big/small, tall/low,
  round/sharp, curved/straight, flared — "long handle", "deep bowl", "sharp tip", "thick bottom", "long stem"
- **materials**: glass, steel, ceramic, porcelain, plastic, wooden — change how the camera sees the object

Anything you do not specify is picked at random within realistic limits.

## Shapes

Containers (mug, cup, glass, wine glass, bowl, plate, frying pan) can take any outline, not just round:

```
python generate.py "square mug"
python generate.py "oval mug with a big handle"
python generate.py "hexagonal glass"          # also triangular, pentagonal, octagonal, "6-sided"
python generate.py "fluted bowl"              # wavy outline; also "wavy", "scalloped", "star", "flower"
python generate.py "twisted square glass"
python generate.py "barrel mug"               # bulging sides; "hourglass" pinches them in
python generate.py "square cup on a saucer"   # the saucer follows the cup's outline
```

Square goes all the way from a rounded square ("slightly square", "squarish") to a hard one; oval from
"slightly oval" to "very oval"/"elliptical".

## Crosses

Two objects can be mixed into one:

```
python generate.py "cross between a mug and a bowl"
python generate.py "70% mug 30% wine glass"
python generate.py "teacup mixed with a wine glass"
python generate.py "half cup half glass"
python generate.py "mug shaped like a bowl"
python generate.py "mostly a glass, a bit of a bowl"
python generate.py "square mug-bowl hybrid"
python generate.py "spork"                    # also "knork" (knife + fork), "splayd" (spoon + knife)
```

Containers cross with containers (their profiles morph into each other; handles, feet and saucers are kept from
whichever object has enough weight), and fork, spoon, knife and spatula cross with each other (their outlines,
thickness and dish morph). Crosses are always built by the formulas, also with `--neural`.

## Options

| flag | effect |
|---|---|
| `--table` | the object lies or stands on a table, and the table is in the point cloud too |
| `--views N` | N cameras shoot the object from different sides; each view is saved and all are fused into `cloud.ply` |
| `--count N` | generate N different objects matching the description (a small dataset), each in `sample_XXX/` |
| `--seed N` | the same seed gives the same result; without it every run is new |
| `--neural` | the trained neural network builds the shape instead of the procedural generator |
| `--wild` | containers also get a random outline (square, oval, wavy, twisted…) unless the text sets one |
| `--no-view` | do not open the browser |
| `--out DIR` | output folder (default `outputs`) |

## How it works

```
text ──► kitchgen/text.py ──► object kind + parameters
                                   │
           ┌───────────────────────┴───────────────────────┐
  kitchgen/shapes.py (SDF from formulas)          kitchgen/neural.py (SDF from a neural network)
           └───────────────────────┬───────────────────────┘
                                   ▼
               kitchgen/mesh.py: marching cubes → mesh
                                   ▼
           kitchgen/stereo.py: virtual stereo camera → point cloud
                                   ▼
              kitchgen/viz.py: browser viewer
```

- **Shapes** are signed distance functions: for any point in space, the distance to the surface, negative inside.
  Cutlery is built from flat 2D outlines extruded into plates and joined with smooth unions; containers are
  profiles revolved around the vertical axis, with ring handles and foot rings. The "radius" a container's
  profile is measured in is not plain round distance: a superellipse norm makes it square, scaling one axis makes it
  oval, a polygon norm gives N sides, a cosine term makes it wavy, and rotating it with height twists it.
- **Crosses** (`kitchgen/hybrid.py`) morph shapes instead of averaging distances (averaging two thin walls that do
  not overlap would leave nothing): container profiles are resampled by arc length and blended point by point, and
  flat cutlery blends its 2D outline, thickness and curve.
- **Meshes** come from marching cubes on a grid of 0.6–1.2 mm depending on the object.
- **The stereo camera** is modelled on an Intel RealSense D435: 1280×720, 5 cm baseline. It sees only its own side of
  the object, makes stereo shadows (a point must be visible to both cameras), adds depth noise that grows as z²,
  rounds disparity to 1/8 pixel, drops points on metal highlights, makes flying pixels on depth edges and random holes.
  Materials matter: metal blinds the camera where it faces it head-on, ceramic and plastic return clean data, and
  glass is mostly invisible — the camera often sees the table straight through it.
- **The neural network** is a conditional SDF: a point (x, y, z) plus a description of the object (kind and
  parameters) goes in, the distance to the surface comes out. It uses Fourier features for thin details,
  feeds the condition into every layer and has a skip connection in the middle. Each point also comes with
  its *warp radius*: the distance from the container's axis measured in the object's own square, oval, wavy or
  twisted outline. A square mug's walls sit at one warp radius all the way round, so the network learns the
  profile, handle and foot instead of having to invent squares and stars from x, y, z. It is trained on
  objects made by the procedural generator, including all those outlines, so `--neural` can draw "square mug"
  too.

## Training the neural network

```
python train.py                   # build the training data (once, ~7 min) and train (~50 min on an RTX 3060)
python train.py --steps 120000    # train longer for crisper prongs and rims
python train.py --rebuild         # rebuild the training data
```

Training prints the mean distance error in millimetres on training objects and on unseen ones.
Afterwards `outputs/neural_check.png` shows each object built by the formulas next to the network's version.

To look inside the trained network:

```
python show_network.py "square mug"             # a window you can rotate with the mouse
python show_network.py "fork" --save net.png    # or a picture
```

Neurons are balls coloured by what they output for one point on the object's surface, lines are the strongest
weights (blue positive, red negative). The top row is the main path (point → Fourier features → 6 hidden
layers → distance); the bottom row turns the object's description into a 64-number code that feeds every layer.

## Install

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
