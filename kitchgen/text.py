import re
from dataclasses import dataclass, field

import numpy as np

from .shapes import DISPLAY_NAMES, KINDS, MOD_NEUTRAL, MOD_RANGES, RANGES

KIND_PATTERNS = [
    ("wine_glass", r"wine ?glass|goblet|champagne|flute|stemware|stemmed glass|martini|coupe"),
    ("pan", r"frying ?pan|fry ?pan|skillet|saut[eé] ?pan|\bpan\b|\bwok\b|crepe pan"),
    ("ladle", r"ladle"),
    ("spatula", r"spatula|turner|flipper"),
    ("mug", r"\bmug|\bstein\b"),
    ("cup", r"tea ?cup|coffee ?cup|espresso|demitasse|\bcups?\b"),
    ("glass", r"\bglass(?:es)?\b|tumbler|highball|lowball|\bpint\b"),
    ("bowl", r"\bbowl"),
    ("plate", r"\bplates?\b|\bdish\b|saucer|platter"),
    ("spoon", r"spoon"),
    ("fork", r"\bforks?\b|trident"),
    ("knife", r"knife|knives|cleaver"),
]

PRESETS = [
    ("spoon", r"tea ?spoon|coffee ?spoon", dict(bowl_len=0.1, bowl_width=0.1, bowl_depth=0.3, handle_len=0.2), {}),
    ("spoon", r"table ?spoon|soup ?spoon|serving ?spoon", dict(bowl_len=0.85, bowl_width=0.8, bowl_depth=0.7, handle_len=0.7), {}),
    ("spoon", r"dessert ?spoon", dict(bowl_len=0.45, bowl_width=0.45, handle_len=0.45), {}),
    ("knife", r"butter ?knife|table ?knife|dinner ?knife", dict(tip_round=1.0, blade_len=0.15, blade_width=0.2), {"serrated": 0}),
    ("knife", r"chef'?s? ?knife|kitchen ?knife|cook'?s? ?knife", dict(blade_len=0.75, blade_width=0.85, tip_round=0.0, handle_len=0.8, handle_thickness=0.7), {"serrated": 0}),
    ("knife", r"bread ?knife", dict(blade_len=0.85, blade_width=0.4, tip_round=0.7), {"serrated": 1}),
    ("knife", r"steak ?knife", dict(blade_len=0.3, blade_width=0.25, tip_round=0.0), {"serrated": 1}),
    ("knife", r"paring ?knife", dict(blade_len=0.02, blade_width=0.15, tip_round=0.0, handle_len=0.2), {"serrated": 0}),
    ("knife", r"cleaver", dict(blade_len=0.6, blade_width=1.0, tip_round=1.0, handle_thickness=0.9), {"serrated": 0}),
    ("fork", r"trident", {}, {"n_tines": 3}),
    ("fork", r"dessert ?fork|cake ?fork|salad ?fork", dict(handle_len=0.25, tine_len=0.3, head_width=0.3), {}),
    ("fork", r"dinner ?fork|table ?fork", dict(handle_len=0.65, tine_len=0.6), {}),
    ("fork", r"serving ?fork|carving ?fork|meat ?fork", dict(handle_len=0.9, tine_len=0.9, head_width=0.2), {"n_tines": 2}),
    ("cup", r"espresso|demitasse", dict(top_radius=0.05, bottom_radius=0.2, height=0.1), {}),
    ("cup", r"tea ?cup", dict(top_radius=0.75, height=0.35, belly=0.7), {}),
    ("cup", r"coffee ?cup", dict(top_radius=0.5, height=0.65, belly=0.4), {}),
    ("mug", r"travel ?mug", dict(height=1.0, radius=0.35, taper=0.85), {}),
    ("mug", r"beer ?mug|stein", dict(height=0.95, radius=0.9, wall=0.9, bottom=0.9), {}),
    ("glass", r"shot ?glass", dict(radius=0.05, height=0.0, bottom=1.0, wall=0.8), {}),
    ("glass", r"\bpint\b", dict(radius=0.9, height=0.95, taper=0.9), {}),
    ("glass", r"highball|collins", dict(radius=0.3, height=0.85, taper=0.1), {}),
    ("glass", r"rocks ?glass|old ?fashioned|whiske?y ?glass|lowball", dict(radius=0.85, height=0.1, bottom=0.9, taper=0.1), {}),
    ("wine_glass", r"champagne|flute", dict(bowl_radius=0.0, bowl_height=0.9, rim_ratio=0.8, stem_height=0.7), {}),
    ("wine_glass", r"red ?wine", dict(bowl_radius=0.9, rim_ratio=0.35), {}),
    ("wine_glass", r"white ?wine", dict(bowl_radius=0.4, rim_ratio=0.55), {}),
    ("wine_glass", r"martini|coupe", dict(bowl_radius=0.9, bowl_height=0.0, rim_ratio=1.0), {}),
    ("bowl", r"soup ?bowl|cereal ?bowl|ramen", dict(height=0.7, roundness=0.7), {}),
    ("bowl", r"salad ?bowl|mixing ?bowl|serving ?bowl", dict(radius=0.95, height=0.6), {}),
    ("bowl", r"rice ?bowl", dict(radius=0.15, height=0.5, roundness=0.8), {}),
    ("plate", r"dinner ?plate", dict(radius=0.85, rim_height=0.3), {}),
    ("plate", r"soup ?plate|pasta ?plate|deep ?plate", dict(radius=0.6, rim_height=0.95, rim_width=0.7), {}),
    ("plate", r"side ?plate|dessert ?plate|bread ?plate|salad ?plate", dict(radius=0.25), {}),
    ("plate", r"saucer", dict(radius=0.0, rim_height=0.2, rim_width=0.6), {}),
    ("plate", r"platter", dict(radius=1.0, rim_height=0.3), {}),
    ("ladle", r"soup ?ladle", dict(bowl_radius=0.7, bowl_depth=0.85), {}),
    ("ladle", r"sauce ?ladle|gravy", dict(bowl_radius=0.1, handle_len=0.2), {}),
    ("spatula", r"slotted", {}, {"slots": 4}),
    ("spatula", r"solid|unslotted", {}, {"slots": 0}),
    ("pan", r"\bwok\b", dict(radius=0.9, depth=1.0, flare=0.0), {}),
    ("pan", r"crepe|cr[eê]pe|pancake", dict(depth=0.0, flare=1.0), {}),
    ("pan", r"saut[eé]", dict(depth=0.9, flare=0.9), {}),
]

NUMBERS = {"one": 1, "single": 1, "two": 2, "double": 2, "three": 3, "triple": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COUNTS = {"n_tines": r"prong|tine|teeth|tooth", "slots": r"slot"}
LENGTH_UNITS = {"mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001,
                "millimetres": 0.001, "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
                "centimetre": 0.01, "centimetres": 0.01, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}
VOLUME_UNITS = {"ml": 1e-6, "milliliter": 1e-6, "milliliters": 1e-6, "millilitre": 1e-6, "millilitres": 1e-6,
                "cl": 1e-5, "l": 1e-3, "liter": 1e-3, "liters": 1e-3, "litre": 1e-3, "litres": 1e-3,
                "oz": 2.957e-5, "ounce": 2.957e-5, "ounces": 2.957e-5}
PARTS = {
    "handle": r"handle|grip|shaft",
    "tines": r"prong|tine|teeth|tooth",
    "blade": r"blade|edge",
    "bowl": r"bowl|head|scoop",
    "tip": r"tip|point",
    "stem": r"stem",
    "foot": r"foot|base|bottom",
    "rim": r"rim|lip|brim",
    "wall": r"wall|side",
}
ADJECTIVES = {
    "long": r"long|lengthy",
    "short": r"short|stubby",
    "wide": r"wide|broad|fat",
    "narrow": r"narrow|slim|slender|skinny",
    "thick": r"thick|chunky|heavy|sturdy",
    "thin": r"thin|delicate",
    "deep": r"deep",
    "shallow": r"shallow|flat",
    "big": r"big|large|huge|giant|oversized|jumbo",
    "small": r"small|little|tiny|mini|petite|kids?|child",
    "tall": r"tall|high",
    "low": r"low|squat",
    "round": r"round|rounded|blunt",
    "sharp": r"sharp|pointed|pointy",
    "curved": r"curved|bent|angled",
    "straight": r"straight",
    "flared": r"flared|tapered|conical|cone",
}
SUFFIX = r"(?:[gnt]?er|[gnt]?est|r|st|ish|ly|ed|d)?"

RULES = {
    ("long", "handle"): [("handle_len", 0.9)],
    ("short", "handle"): [("handle_len", 0.1)],
    ("thick", "handle"): [("handle_width", 0.85), ("handle_thickness", 0.9), ("handle_radius", 0.9)],
    ("thin", "handle"): [("handle_width", 0.1), ("handle_thickness", 0.1), ("handle_radius", 0.1)],
    ("wide", "handle"): [("handle_width", 0.9)],
    ("narrow", "handle"): [("handle_width", 0.1)],
    ("big", "handle"): [("handle_size", 0.9), ("handle_len", 0.8), ("handle_width", 0.7)],
    ("small", "handle"): [("handle_size", 0.1), ("handle_len", 0.2), ("handle_width", 0.3)],
    ("curved", "handle"): [("bend", 0.9), ("handle_angle", 0.9)],
    ("straight", "handle"): [("bend", 0.05), ("handle_angle", 0.05)],
    ("long", "tines"): [("tine_len", 0.9)],
    ("short", "tines"): [("tine_len", 0.1)],
    ("long", "blade"): [("blade_len", 0.9)],
    ("short", "blade"): [("blade_len", 0.1)],
    ("wide", "blade"): [("blade_width", 0.9)],
    ("narrow", "blade"): [("blade_width", 0.1)],
    ("thick", "blade"): [("blade_thickness", 0.9)],
    ("thin", "blade"): [("blade_thickness", 0.1)],
    ("big", "blade"): [("blade_len", 0.8), ("blade_width", 0.8)],
    ("small", "blade"): [("blade_len", 0.2), ("blade_width", 0.2)],
    ("wide", "bowl"): [("bowl_width", 0.9), ("head_width", 0.9), ("bowl_radius", 0.85)],
    ("narrow", "bowl"): [("bowl_width", 0.1), ("head_width", 0.1), ("bowl_radius", 0.15)],
    ("deep", "bowl"): [("bowl_depth", 0.9)],
    ("shallow", "bowl"): [("bowl_depth", 0.1)],
    ("big", "bowl"): [("bowl_len", 0.9), ("bowl_width", 0.85), ("bowl_radius", 0.9)],
    ("small", "bowl"): [("bowl_len", 0.1), ("bowl_width", 0.15), ("bowl_radius", 0.1)],
    ("sharp", "tip"): [("tip_round", 0.0)],
    ("round", "tip"): [("tip_round", 1.0)],
    ("long", "stem"): [("stem_height", 0.9)],
    ("tall", "stem"): [("stem_height", 0.9)],
    ("short", "stem"): [("stem_height", 0.1)],
    ("low", "stem"): [("stem_height", 0.1)],
    ("thick", "stem"): [("stem_radius", 0.9)],
    ("thin", "stem"): [("stem_radius", 0.1)],
    ("wide", "foot"): [("foot_radius", 0.9)],
    ("big", "foot"): [("foot_radius", 0.9)],
    ("narrow", "foot"): [("foot_radius", 0.1)],
    ("small", "foot"): [("foot_radius", 0.1)],
    ("thick", "foot"): [("bottom", 0.9), ("foot_height", 0.9)],
    ("thick", "wall"): [("wall", 0.9)],
    ("thin", "wall"): [("wall", 0.1)],
    ("wide", "rim"): [("rim_width", 0.9)],
    ("narrow", "rim"): [("rim_width", 0.1)],
    ("tall", "rim"): [("rim_height", 0.9)],
    ("low", "rim"): [("rim_height", 0.1)],
    ("long", None): [("handle_len", 0.85), ("tine_len", 0.8), ("blade_len", 0.85), ("bowl_len", 0.7)],
    ("short", None): [("handle_len", 0.15), ("tine_len", 0.2), ("blade_len", 0.15), ("bowl_len", 0.3),
                      ("height", 0.1), ("stem_height", 0.1), ("depth", 0.1)],
    ("tall", None): [("height", 0.9), ("stem_height", 0.85), ("bowl_height", 0.8), ("depth", 0.9)],
    ("low", None): [("height", 0.1), ("stem_height", 0.15), ("bowl_height", 0.2), ("depth", 0.1), ("rim_height", 0.1)],
    ("wide", None): [("head_width", 0.9), ("bowl_width", 0.9), ("blade_width", 0.9), ("handle_width", 0.7),
                     ("radius", 0.9), ("top_radius", 0.9), ("bowl_radius", 0.85), ("foot_radius", 0.7)],
    ("narrow", None): [("head_width", 0.1), ("bowl_width", 0.1), ("blade_width", 0.1), ("handle_width", 0.3),
                       ("radius", 0.1), ("top_radius", 0.1), ("bowl_radius", 0.15)],
    ("thick", None): [("thickness", 0.9), ("wall", 0.9), ("bottom", 0.85), ("handle_thickness", 0.8),
                      ("blade_thickness", 0.8), ("handle_radius", 0.8)],
    ("thin", None): [("thickness", 0.1), ("wall", 0.1), ("bottom", 0.15), ("blade_thickness", 0.1),
                     ("handle_radius", 0.15), ("stem_radius", 0.1)],
    ("deep", None): [("bowl_depth", 0.9), ("depth", 0.9), ("height", 0.85), ("rim_height", 0.9)],
    ("shallow", None): [("bowl_depth", 0.1), ("depth", 0.1), ("height", 0.15), ("rim_height", 0.05)],
    ("big", None): [("handle_len", 0.8), ("bowl_len", 0.9), ("bowl_width", 0.85), ("blade_len", 0.8),
                    ("tine_len", 0.75), ("head_width", 0.75), ("radius", 0.85), ("height", 0.8),
                    ("top_radius", 0.8), ("bowl_radius", 0.85), ("bowl_height", 0.8), ("blade_width", 0.75)],
    ("small", None): [("handle_len", 0.2), ("bowl_len", 0.1), ("bowl_width", 0.15), ("bowl_depth", 0.3),
                      ("blade_len", 0.2), ("tine_len", 0.25), ("head_width", 0.25), ("radius", 0.15),
                      ("height", 0.15), ("top_radius", 0.15), ("bottom_radius", 0.2), ("bowl_radius", 0.15),
                      ("bowl_height", 0.2), ("blade_width", 0.25), ("saucer_radius", 0.2)],
    ("round", None): [("tip_round", 1.0), ("roundness", 1.0), ("belly", 0.9)],
    ("sharp", None): [("tip_round", 0.0)],
    ("curved", None): [("bend", 0.9), ("handle_angle", 0.85), ("belly", 0.8)],
    ("straight", None): [("bend", 0.05), ("handle_angle", 0.1), ("belly", 0.05)],
    ("flared", None): [("taper", 0.95), ("flare", 0.1), ("belly", 0.1), ("rim_ratio", 1.0), ("top_radius", 0.9)],
}
LENGTH_TARGETS = {"handle": "handle_len", "blade": "blade_len", "tines": "tine_len", "stem": "stem_height"}
SHAPE_WORDS = [
    (r"squarish|rounded square|soft square", dict(squareness=0.55), "rounded square"),
    (r"\bsquare|\bcub(?:e|ic|oid)\b|\bboxy\b|\bblocky\b", dict(squareness=1.0), "square"),
    (r"\boval|ellip|elongated|egg.shaped", dict(aspect=0.68), "oval"),
    (r"triang", dict(sides=3), "triangular"),
    (r"pentag", dict(sides=5), "pentagonal"),
    (r"hexag|six.sided", dict(sides=6), "hexagonal"),
    (r"heptag|seven.sided", dict(sides=7), "heptagonal"),
    (r"octag|eight.sided", dict(sides=8), "octagonal"),
    (r"fluted|ribbed|faceted|grooved", dict(lobes=14, lobe_depth=0.05), "fluted"),
    (r"scallop|wavy|flower|petal|lotus", dict(lobes=7, lobe_depth=0.12), "wavy"),
    (r"\bstar", dict(lobes=5, lobe_depth=0.25), "star"),
    (r"twist|spiral|swirl", dict(twist=2.0), "twisted"),
    (r"barrel|bulg|bellied|\bbelly\b|tub.shaped", dict(bulge=0.2), "barrel belly"),
    (r"hourglass|waisted|pinched|cinched|curvy", dict(bulge=-0.15), "hourglass waist"),
]
SHAPE_STRENGTH = [(r"(?:slightly|a bit|a little|somewhat|mildly|gently|kind of|sort of)[ -]$", 0.45, "slightly "),
                  (r"(?:very|extremely|really|super|strongly|heavily|deeply|totally|perfectly)[ -]$", 1.5, "very ")]
NAMED_CROSSES = [(r"\bsporks?\b", "spoon", "fork"), (r"\bknorks?\b", "knife", "fork"), (r"\bsplayds?\b", "spoon", "knife")]
CROSS_PATTERNS = [
    r"(?:cross|hybrid|mix(?:ture)?|blend|mash.?up|combo|combination|merge|morph|fusion)\s+(?:between|of|from)\s+"
    r"(?P<a>.+?)\s+(?:and|with|&|\+)\s+(?P<b>.+)",
    r"(?P<a>.+?)\s+(?:crossed|mixed|merged|combined|blended|fused|morphed|hybridi[sz]ed)\s+(?:with|and|into)\s+"
    r"(?P<b>.+)",
    r"half[ -](?P<a>.+?)[, ]+(?:and\s+)?half[ -](?P<b>.+)",
    r"(?P<a>.+?)\s+(?:shaped|that looks|looking|styled)\s+like\s+(?P<b>.+)",
    r"(?P<a>.+?)\s*(?:\+|\bx\b|\bmeets\b)\s*(?P<b>.+)",
    r"(?P<a>[a-z' ]+?)[ -](?P<b>[a-z' ]+?)\s+(?:hybrid|cross|mix|mashup)",
    r"(?P<a>\d+\s*%\s*[a-z' ]+?)\s+(?:and\s+)?(?P<b>\d+\s*%\s*[a-z' ]+)",
    r"mostly\s+(?P<a>.+?)[,;]?\s+(?:and\s+|with\s+|but\s+)?(?:a bit|slightly|a little|a hint|a touch)\s+(?P<b>.+)",
]
MATERIAL_WORDS = {
    "glass": r"\bglass\b|crystal",
    "metal": r"steel|metal|stainless|alumin(?:i)?um|copper|cast ?iron|\biron\b|silver|titanium",
    "ceramic": r"ceramic|porcelain|\bchina\b|stoneware|earthenware|clay|enamel",
    "plastic": r"plastic|silicone|nylon|wood(?:en)?|bamboo|melamine",
}
DEFAULT_SIZE = {"fork": "length", "spoon": "length", "knife": "blade_len", "ladle": "length",
                "spatula": "length", "mug": "height", "cup": "height", "glass": "height",
                "wine_glass": "height", "bowl": "diameter", "plate": "diameter", "pan": "diameter"}


@dataclass
class Request:
    kind: str | None
    fractions: dict = field(default_factory=dict)
    exact: dict = field(default_factory=dict)
    length: float | None = None
    height: float | None = None
    diameter: float | None = None
    capacity: float | None = None
    material: str | None = None
    cross_kind: str | None = None
    cross_weight: float = 0.5
    notes: list = field(default_factory=list)

    def fixed_params(self, rng, jitter=0.06):
        out = {}
        for name, frac in self.fractions.items():
            ranges = RANGES[self.kind] if name in RANGES[self.kind] else RANGES[self.cross_kind]
            lo, hi = ranges[name]
            frac = float(np.clip(frac + rng.uniform(-jitter, jitter), 0.0, 1.0))
            out[name] = lo + frac * (hi - lo)
        out.update(self.exact)
        return out


def _number(word):
    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return float(word)
    return NUMBERS.get(word)


def _part_at(word):
    for part, pat in PARTS.items():
        if re.match(pat, word):
            return part
    return None


def _find_part(words, i, ahead=3, behind=2):
    for j in list(range(i + 1, min(i + 1 + ahead, len(words)))) + list(range(i - 1, max(i - 1 - behind, -1), -1)):
        part = _part_at(words[j])
        if part:
            return part, words[j]
    return None, None


def _find_kind(low):
    spans = []
    for kind, pat in KIND_PATTERNS:
        spans += [(m.start(), m.end(), kind) for m in re.finditer(pat, low)]
    spans = [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and o != s and (o[1] - o[0]) > (s[1] - s[0])
                                         for o in spans)]
    if not spans:
        return None
    m = re.search(r"\b(?:with|without|and|on|having|that|for|of|in|plus|w/o?)\b", low)
    head = [s for s in spans if m is None or s[1] <= m.start()]
    return max(head)[2] if head else min(spans)[2]


def _apply_rules(req, rules, allowed):
    hit = False
    for name, frac in rules:
        if name in allowed and name not in req.exact:
            req.fractions[name] = frac
            hit = True
    return hit


def _find_cross(low):
    for pat, a, b in NAMED_CROSSES:
        if re.search(pat, low):
            return a, b, 0.5
    for pat in CROSS_PATTERNS:
        m = re.search(pat, low)
        if not m:
            continue
        a, b = _find_kind(m.group("a")), _find_kind(m.group("b"))
        if a and b and a != b:
            w = 0.5
            for pct, words in re.findall(r"(\d+)\s*%\s*(?:of\s+)?(?:a\s+|an\s+)?([a-z' ]+?)(?=\s*(?:\d|,|and|$))", low):
                k = _find_kind(words)
                if k == b:
                    w = float(pct) / 100
                elif k == a:
                    w = 1 - float(pct) / 100
            if re.search(r"mostly\s+(?:a\s+|an\s+)?" + re.escape(m.group("a").split()[-1]), low):
                w = 0.3
            elif re.search(r"(?:mostly|more)\s+(?:of\s+)?(?:a\s+|an\s+)?" + re.escape(m.group("b").split()[-1]), low):
                w = 0.7
            elif re.search(r"(?:a bit|slightly|a little|hint)\s+(?:of\s+)?(?:a\s+|an\s+)?" + re.escape(m.group("b").split()[-1]), low):
                w = 0.25
            return a, b, float(np.clip(w, 0.05, 0.95))
    return None


def _flags(low, kind, req):
    def has(pattern):
        return re.search(pattern, low) is not None
    no = r"(?:without|no|w/o|sans)\s+(?:a\s+|an\s+|the\s+)?"
    yes = r"(?:with|w/|has|having|plus|and|on)\s+(?:a\s+|an\s+|the\s+)?"
    if kind in ("mug", "cup"):
        if has(no + r"handles?") or has(r"handleless"):
            req.exact["has_handle"] = 0
            req.notes.append("no handle")
        elif has(yes + r"(?:\w+\s+)?handles?"):
            req.exact["has_handle"] = 1
    if kind == "cup":
        if has(no + r"saucers?"):
            req.exact["saucer"] = 0
            req.notes.append("no saucer")
        elif has(r"saucer"):
            req.exact["saucer"] = 1
            req.notes.append("on a saucer")
    if kind == "ladle":
        if has(no + r"hooks?"):
            req.exact["hook"] = 0
            req.notes.append("no hook")
        elif has(r"hook"):
            req.exact["hook"] = 1
            req.notes.append("hooked end")
    if kind == "knife":
        if has(r"serrat|toothed|jagged"):
            req.exact["serrated"] = 1
            req.notes.append("serrated edge")
        elif has(r"smooth|plain (?:edge|blade)|straight edge"):
            req.exact["serrated"] = 0
    if kind == "spatula" and "slots" not in req.exact:
        if has(no + r"slots?"):
            req.exact["slots"] = 0
            req.notes.append("no slots")


def _apply_size(req, kind, words, i, value, allowed):
    near = words[max(0, i - 4): i + 5]
    part = next((p for p in map(_part_at, near) if p), None)
    target = None
    if part in LENGTH_TARGETS and LENGTH_TARGETS[part] in allowed:
        target = LENGTH_TARGETS[part]
    elif any(re.match(r"tall|high|height|deep|depth", w) for w in near):
        target = "height"
    elif any(re.match(r"wide|width|diameter|across|dia\b", w) for w in near):
        target = "diameter"
    elif any(re.match(r"long|length", w) for w in near):
        target = "length"
    else:
        target = DEFAULT_SIZE[kind]
    if target in ("length", "height", "diameter"):
        setattr(req, target, value)
        req.notes.append(f"{target} {value * 100:.1f} cm")
    else:
        req.exact[target] = value
        req.notes.append(f"{target.replace('_', ' ')} {value * 100:.1f} cm")


def parse(text):
    low = text.lower().replace("″", " inch ").replace('"', " inch ")
    low = re.sub(r"(\d)([a-z])", r"\1 \2", low)
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", low)

    crossing = _find_cross(low)
    kind = crossing[0] if crossing else _find_kind(low)
    req = Request(kind=kind)
    if kind is None:
        return req
    if crossing:
        req.cross_kind, req.cross_weight = crossing[1], crossing[2]
        req.notes.append(f"cross with a {DISPLAY_NAMES[crossing[1]]} ({100 * (1 - crossing[2]):.0f}/"
                         f"{100 * crossing[2]:.0f})")
    allowed = set(RANGES[kind])
    if crossing:
        allowed |= set(RANGES[crossing[1]])

    for pat, values, label in SHAPE_WORDS:
        m = re.search(pat, low)
        if not m or not any(k in allowed for k in values) or any(k in req.exact for k in values):
            continue
        strength, how = 1.0, ""
        for spat, s, word in SHAPE_STRENGTH:
            if re.search(spat, low[:m.start()]):
                strength, how = s, word
        for k, v in values.items():
            if isinstance(v, float):
                lo, hi = MOD_RANGES[k]
                v = float(np.clip(MOD_NEUTRAL[k] + strength * (v - MOD_NEUTRAL[k]), lo, hi))
            req.exact[k] = v
        req.notes.append(f"{how}{label} shape")
    m = re.search(r"(\d+|" + "|".join(NUMBERS) + r")[ -]sided", low)
    if m:
        req.exact["sides"] = int(_number(m.group(1)))
        req.exact.pop("squareness", None)
        req.notes.append(f"{req.exact['sides']} sides")
    if req.exact.get("twist") and not (req.exact.get("squareness") or req.exact.get("sides") or req.exact.get("lobes")):
        req.exact.update(lobes=6, lobe_depth=0.08)

    for k, pat, fractions, exact in PRESETS:
        m = re.search(pat, low) if k == kind else None
        if m:
            req.fractions.update(fractions)
            req.exact.update(exact)
            req.notes.append(f"preset '{m.group(0)}'")
    _flags(low, kind, req)
    for material, pat in MATERIAL_WORDS.items():
        if material == "glass" and {kind, req.cross_kind} & {"glass", "wine_glass"}:
            continue
        if re.search(pat, low):
            req.material = material
            req.notes.append(f"{material} material")
            break

    for i, w in enumerate(words):
        n = _number(w)
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if n is not None and nxt in LENGTH_UNITS:
            _apply_size(req, kind, words, i, n * LENGTH_UNITS[nxt], allowed)
        elif n is not None and nxt in VOLUME_UNITS:
            req.capacity = n * VOLUME_UNITS[nxt]
            req.notes.append(f"capacity {n * VOLUME_UNITS[nxt] * 1e6:.0f} ml")
        elif n is not None:
            for name, pat in COUNTS.items():
                if name in allowed and any(re.match(pat, x) for x in words[i + 1:i + 3]):
                    req.exact[name] = int(n)
                    req.notes.append(f"{int(n)} {'prongs' if name == 'n_tines' else 'slots'}")

        adj = next((a for a, pat in ADJECTIVES.items() if re.fullmatch(f"(?:{pat}){SUFFIX}", w)), None)
        if adj:
            if adj == "straight" and kind in ("mug", "glass"):
                req.exact["taper"] = 1.0
            part, part_word = _find_part(words, i)
            rules = RULES.get((adj, part), [])
            if not _apply_rules(req, rules, allowed):
                part = None
                _apply_rules(req, RULES.get((adj, None), []), allowed)
            req.notes.append(f"{w} {part_word}" if part else w)

    for name in ("n_tines", "slots"):
        if name in req.exact:
            lo, hi = RANGES[kind][name]
            if not lo <= req.exact[name] <= hi:
                req.exact[name] = int(np.clip(req.exact[name], lo, hi))
                req.notes.append(f"{name.replace('n_', '')} must be {lo}-{hi}, using {req.exact[name]}")
    if any(re.match(r"slotted", w) for w in words) and kind == "spatula" and "slots" not in req.exact:
        req.exact["slots"] = 4
    req.exact = {k: v for k, v in req.exact.items() if k in allowed}
    return req


def describe_kinds():
    return ", ".join(DISPLAY_NAMES[k] for k in KINDS)
