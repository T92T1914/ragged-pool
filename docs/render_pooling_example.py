"""Render the conceptual pooling diagram on Obscur without running the operator.

The diagram is a title, a subtitle, three numbered cards and a footer, in a
wide edition and a narrow edition. Each edition is written as PNG and as SVG.
The SVG draws every letter as an outline, so it needs no installed font.

Rendering needs Pillow, fonttools, uharfbuzz and the static Inter Regular,
SemiBold and Bold TTF files, which are not distributed here:

    python docs/render_pooling_example.py --font-dir <folder with Inter TTFs>

Add --check to compare against the committed files without writing them.
Run with --check-receipt and nothing else to recompute the hashes in
ragged-pool-rendering.json with the standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from html import escape
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "ragged-pool-rendering.json"
TOKEN_SOURCE = (
    "https://github.com/T92T1914/clair-obscur-themes/blob/"
    "4693831b51f2b3fa268d830fd9b977fb70c83e3d/tokens-monochrome.json"
)
# The Obscur values a figure may use. No other color is drawn.
TOKENS = {
    "canvas": "#000000",
    "control": "#151515",
    "text": "#ffffff",
    "muted": "#cccccc",
    "border": "#999999",
}
FACES = (("Regular", 400), ("SemiBold", 600), ("Bold", 700))
POINT = 100 / 72  # Sizes are points at 100 dpi.
LINE = 1.3
SUPERSAMPLE = 4
# Kerning only. Ligatures and contextual forms stay off, so one letter is one glyph.
FEATURES = {"kern": True, "liga": False, "calt": False, "clig": False}

TITLE = "Ragged Pooling for PyTorch"
SUBTITLE = "One summary vector per group"
CARDS = (
    ("1. Groups", ("Two items, three items or none.",
                   "Each item has a value and a score.")),
    ("2. Softmax", ("Scores in a group become weights.",
                    "An empty group has no weights.")),
    ("3. Weighted sum", ("One vector per group.",
                         "An empty group gives zeros.",
                         "Gradients reach values and scores.")),
)
FOOTER = ("CPU layer in C++ for PyTorch, with packed and padded inputs and "
          "first order gradients. A conceptual example, not a benchmark or a model "
          "quality result.")
DESCRIPTION = (
    "Groups with two items, three items or none. Softmax turns the scores in "
    "each nonempty group into weights. A weighted sum gives one vector per "
    "group, and an empty group gives zeros. Gradients reach values and scores. "
    "A conceptual example of the CPU layer, not a benchmark or a model quality "
    "result."
)
EDITIONS = {
    "wide": {
        "name": "ragged-pool-obscur", "width": 1800, "min_height": 800,
        "direction": "row", "arrow_head": (14, 8),
        "margin": 72, "card_width": 500, "card_gap": 78,
        "title": 40, "subtitle": 30, "card_title": 32, "body": 30, "footer": 30,
    },
    "narrow": {
        "name": "ragged-pool-obscur-narrow", "width": 960, "min_height": 0,
        "direction": "column", "arrow_head": (10, 6),
        "margin": 48, "card_width": 864, "card_gap": 24,
        "title": 40, "subtitle": 33, "card_title": 34, "body": 33, "footer": 33,
    },
}
PAD_SIDE, PAD_TOP, PAD_BOTTOM = 28, 32, 28
BODY_GAP, TITLE_GAP = 14, 20
RADIUS, OUTLINE, CONNECTOR = 18, 2, 2.5
INSET = 12
AFTER_TITLE, AFTER_SUBTITLE, BEFORE_FOOTER, AFTER_FOOTER = 8, 48, 48, 49


def sha256_lf(path: Path) -> str:
    """Hash a text file with LF line endings, so a Windows checkout agrees."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_output(path: Path) -> str:
    return sha256_lf(path) if path.suffix == ".svg" else hashlib.sha256(path.read_bytes()).hexdigest()


class Face:
    """One static Inter face, checked before anything is drawn."""

    def __init__(self, path: Path, name: str, weight: int):
        from fontTools.ttLib import TTFont

        self.path, self.weight = path, weight
        self.font = TTFont(path)
        cmap = self.font.getBestCmap()
        if (self.font["name"].getDebugName(6) != f"Inter-{name}"
                or self.font["OS/2"].usWeightClass != weight
                or self.font["OS/2"].fsSelection & 1
                or "fvar" in self.font
                or not set(range(32, 127)).issubset(cmap)):
            raise ValueError(f"Not a static upright Inter {name} face with Latin coverage: {path.name}")
        self.cmap = cmap
        self.glyphs = self.font.getGlyphSet()
        self.upem = self.font["head"].unitsPerEm
        self.ascent = self.font["hhea"].ascent
        self.descent = self.font["hhea"].descent
        import uharfbuzz as hb

        self.hb = hb.Font(hb.Face(hb.Blob(path.read_bytes())))
        self.outline, self.bounds = {}, {}

    def shape(self, text: str):
        """Return each letter with its glyph name and pen position in font units."""
        import uharfbuzz as hb

        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        hb.shape(self.hb, buffer, FEATURES)
        infos, positions = buffer.glyph_infos, buffer.glyph_positions
        if [info.cluster for info in infos] != list(range(len(text))):
            raise ValueError(f"Shaping merged or split letters: {text}")
        placed, pen = [], 0
        for ch, info, position in zip(text, infos, positions):
            name = self.font.getGlyphName(info.codepoint)
            if name != self.cmap[ord(ch)]:
                # The PNG draws letters, so a substituted glyph would differ from the SVG.
                raise ValueError(f"Shaping substituted a glyph for {ch!r} in: {text}")
            placed.append((ch, name, pen + position.x_offset))
            pen += position.x_advance
        return placed, pen

    def width(self, text: str, px: float) -> float:
        return self.shape(text)[1] * px / self.upem

    def glyph(self, name: str):
        """Return the SVG path data and the bounds of one glyph in font units."""
        if name not in self.outline:
            from fontTools.pens.boundsPen import BoundsPen
            from fontTools.pens.svgPathPen import SVGPathPen

            pen = SVGPathPen(self.glyphs)
            self.glyphs[name].draw(pen)
            bounds = BoundsPen(self.glyphs)
            self.glyphs[name].draw(bounds)
            self.outline[name], self.bounds[name] = pen.getCommands(), bounds.bounds
        return self.outline[name], self.bounds[name]


def wrap(face: Face, text: str, px: float, width: float) -> list[str]:
    """Wrap by measured width. Type never shrinks to fit."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and face.width(trial, px) > width:
            lines.append(current)
            current = word
        else:
            current = trial
        if face.width(current, px) > width:
            raise ValueError(f"A word does not fit, change the layout: {word}")
    return lines + [current]


class Canvas:
    """Draw the same marks into an SVG and a supersampled PNG."""

    def __init__(self, width: int, height: int):
        from PIL import Image, ImageDraw

        self.width, self.height = width, height
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(TITLE)}</title>',
            f'<desc id="desc">{escape(DESCRIPTION)}</desc>',
            f'<!-- Obscur tokens: {TOKEN_SOURCE}. Text is drawn as outlines. -->',
            f'<rect width="{width}" height="{height}" fill="{TOKENS["canvas"]}"/>',
        ]
        self.image = Image.new("RGB", (width*SUPERSAMPLE, height*SUPERSAMPLE), TOKENS["canvas"])
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.labels = 0

    def card(self, x, y, w, h):
        # Pillow draws the outline inside the box, so the SVG stroke is inset to match.
        half = OUTLINE / 2
        self.svg.append(f'<rect x="{x + half}" y="{y + half:.2f}" width="{w - OUTLINE}" '
                        f'height="{h - OUTLINE:.2f}" rx="{RADIUS - half}" '
                        f'fill="{TOKENS["control"]}" stroke="{TOKENS["border"]}" '
                        f'stroke-width="{OUTLINE}"/>')
        s = SUPERSAMPLE
        self.draw.rounded_rectangle((x*s, y*s, (x+w)*s, (y+h)*s), RADIUS*s,
                                    TOKENS["control"], TOKENS["border"], OUTLINE*s)

    def arrow(self, x1, y1, x2, y2, head, half):
        """A connector in text with a filled head. It shows direction only."""
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        bx, by = x2 - ux*head, y2 - uy*head
        points = [(x2, y2), (bx - uy*half, by + ux*half), (bx + uy*half, by - ux*half)]
        self.svg.append(f'<path d="M {x1:.2f} {y1:.2f} L {bx:.2f} {by:.2f}" '
                        f'stroke="{TOKENS["text"]}" stroke-width="{CONNECTOR}" fill="none"/>')
        self.svg.append('<path d="M ' + " L ".join(f"{px:.2f} {py:.2f}" for px, py in points)
                        + f' Z" fill="{TOKENS["text"]}"/>')
        s = SUPERSAMPLE
        self.draw.line((x1*s, y1*s, bx*s, by*s), TOKENS["text"], round(CONNECTOR*s))
        self.draw.polygon([(px*s, py*s) for px, py in points], TOKENS["text"])

    def text(self, face: Face, x: float, baseline: float, value: str, px: float, role: str):
        """Place glyphs by their advances, as outlines in SVG and pixels in PNG."""
        from PIL import ImageFont

        color, scale = TOKENS[role], px / face.upem
        key = (face.weight, px)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(face.path), px*SUPERSAMPLE,
                                                 layout_engine=ImageFont.Layout.BASIC)
        paths, box = [], [math.inf, math.inf, -math.inf, -math.inf]
        placed, _ = face.shape(value)
        for ch, name, pen_x in placed:
            outline, bounds = face.glyph(name)
            if outline:
                paths.append(f'<path transform="translate({pen_x})" d="{outline}"/>')
            if bounds:
                gx = x + pen_x*scale
                box = [min(box[0], gx + bounds[0]*scale), min(box[1], baseline - bounds[3]*scale),
                       max(box[2], gx + bounds[2]*scale), max(box[3], baseline - bounds[1]*scale)]
            self.draw.text(((x + pen_x*scale)*SUPERSAMPLE, baseline*SUPERSAMPLE), ch,
                           color, font=self.fonts[key], anchor="ls")
        self.svg.append(f'<g fill="{color}" transform="translate({x:.2f} {baseline:.2f}) '
                        f'scale({scale:.6f} {-scale:.6f})">' + "".join(paths) + "</g>")
        self.labels += 1
        return box

    def finish(self):
        from PIL import Image

        svg = ("\n".join(self.svg + ["</svg>"]) + "\n").encode("utf-8")
        stream = BytesIO()
        self.image.reduce(SUPERSAMPLE).save(stream, "PNG", optimize=False, compress_level=9)
        return svg, stream.getvalue()


def inside(box, left, top, right, bottom, what):
    if box[0] < left or box[1] < top or box[2] > right or box[3] > bottom:
        raise ValueError(f"A label leaves its {what}: {box}")


def layout(edition: dict, faces: dict):
    """Measure every line first, so card and canvas heights come from the text."""
    regular, semibold, bold = faces[400], faces[600], faces[700]
    width, margin, card_w = edition["width"], edition["margin"], edition["card_width"]
    inner = card_w - 2*PAD_SIDE
    px = {role: edition[role]*POINT for role in ("title", "subtitle", "card_title", "body", "footer")}
    title = wrap(bold, TITLE, px["title"], width - 2*margin)
    subtitle = wrap(regular, SUBTITLE, px["subtitle"], width - 2*margin)
    if len(subtitle) != 1:
        raise ValueError("The subtitle must stay on one line")
    cards = [(wrap(semibold, head, px["card_title"], inner),
              [wrap(regular, line, px["body"], inner) for line in body]) for head, body in CARDS]
    footer = wrap(semibold, FOOTER, px["footer"], width - 2*margin)

    def block(lines, size):
        return len(lines) * size * LINE

    def card_height(head, body):
        return (PAD_TOP + block(head, px["card_title"]) + TITLE_GAP
                + sum(block(lines, px["body"]) for lines in body)
                + BODY_GAP*(len(body) - 1) + PAD_BOTTOM)

    heights = [card_height(head, body) for head, body in cards]
    wide = edition["direction"] == "row"
    if wide and card_w*len(cards) + edition["card_gap"]*(len(cards) - 1) + 2*margin != width:
        raise ValueError("A row of cards must fill the width between the margins")
    top = margin + block(title, px["title"]) + AFTER_TITLE + block(subtitle, px["subtitle"]) + AFTER_SUBTITLE
    if wide:
        card_h = max(heights)
        boxes = [(margin + i*(card_w + edition["card_gap"]), top, card_w, card_h) for i in range(len(cards))]
        cards_bottom = top + card_h
    else:
        boxes, y = [], top
        for h in heights:
            boxes.append((margin, y, card_w, h))
            y += h + edition["card_gap"]
        cards_bottom = y - edition["card_gap"]
    footer_top = cards_bottom + BEFORE_FOOTER
    height = max(edition["min_height"], math.ceil(footer_top + block(footer, px["footer"]) + AFTER_FOOTER))
    return {"px": px, "title": title, "subtitle": subtitle, "cards": cards, "footer": footer,
            "boxes": boxes, "wide": wide, "top": margin, "footer_top": footer_top, "height": height}


def render(edition: dict, faces: dict):
    plan = layout(edition, faces)
    px, margin, width, height = plan["px"], edition["margin"], edition["width"], plan["height"]
    canvas = Canvas(width, height)

    def lines(face, values, size, x, top, role, limits):
        for value in values:
            baseline = top + (size*LINE - (face.ascent - face.descent)*size/face.upem)/2 + face.ascent*size/face.upem
            inside(canvas.text(face, x, baseline, value, size, role), *limits)
            top += size*LINE
        return top

    page = (0, 0, width, height, "canvas")
    y = lines(faces[700], plan["title"], px["title"], margin, plan["top"], "text", page)
    lines(faces[400], plan["subtitle"], px["subtitle"], margin, y + AFTER_TITLE, "muted", page)
    for (x, top, w, h), (head, body) in zip(plan["boxes"], plan["cards"]):
        canvas.card(x, top, w, h)
        limits = (x + INSET, top + INSET, x + w - INSET, top + h - INSET, "card")
        y = lines(faces[600], head, px["card_title"], x + PAD_SIDE, top + PAD_TOP, "text", limits)
        y += TITLE_GAP
        for index, statement in enumerate(body):
            y = lines(faces[400], statement, px["body"], x + PAD_SIDE, y, "text", limits)
            y += BODY_GAP if index < len(body) - 1 else 0
    for (x1, y1, w1, h1), (x2, y2, _, _) in zip(plan["boxes"], plan["boxes"][1:]):
        if plan["wide"]:
            mid = y1 + h1/2
            canvas.arrow(x1 + w1 + 10, mid, x2 - 10, mid, *edition["arrow_head"])
        else:
            canvas.arrow(x1 + w1/2, y1 + h1 + 3, x2 + w1/2, y2 - 3, *edition["arrow_head"])
    lines(faces[600], plan["footer"], px["footer"], margin, plan["footer_top"], "muted", page)
    svg, png = canvas.finish()
    record = {
        "width": width, "height": height, "labels_within_canvas": canvas.labels,
        "rendered_card_lines": [[line for statement in body for line in statement] for _, body in plan["cards"]],
        "height_rule": "wrapping by measured width, 1.3 line spacing and fixed padding and gaps",
        "minimum_font_points": min(edition[role] for role in ("subtitle", "body", "footer")),
    }
    return svg, png, record


def load_faces(directory: Path) -> dict:
    return {weight: Face(directory / f"Inter-{name}.ttf", name, weight) for name, weight in FACES}


def build(directory: Path):
    from PIL import __version__ as pillow
    from fontTools import version as fonttools
    from uharfbuzz import __version__ as harfbuzz

    faces = load_faces(directory)
    outputs, layouts = {}, {}
    for key, edition in EDITIONS.items():
        svg, png, layouts[key] = render(edition, faces)
        outputs[f"{edition['name']}.png"] = png
        outputs[f"{edition['name']}.svg"] = svg
    receipt = {
        "schema_version": 1,
        "kind": "conceptual diagram",
        "application_or_scientific_execution": False,
        "tokens": {"source": TOKEN_SOURCE, "values": TOKENS},
        "inputs_sha256_lf": {"docs/render_pooling_example.py": sha256_lf(Path(__file__))},
        "fonts": {name: {"filename": f"Inter-{name}.ttf", "weight": weight,
                         "sha256": hashlib.sha256(faces[weight].path.read_bytes()).hexdigest()}
                  for name, weight in FACES},
        "renderer": {"pillow": pillow, "fonttools": fonttools, "uharfbuzz": harfbuzz,
                     "features": FEATURES, "supersample": SUPERSAMPLE},
        "layouts": layouts,
        "outputs_sha256": {f"docs/{name}": hashlib.sha256(
                               data.replace(b"\r\n", b"\n") if name.endswith(".svg") else data
                           ).hexdigest() for name, data in sorted(outputs.items())},
    }
    outputs[RECEIPT.name] = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    return outputs


def check_receipt() -> list[str]:
    """Recompute every hash the receipt names, with the standard library only."""
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    problems = []
    for name, expected in receipt["inputs_sha256_lf"].items():
        path = ROOT.parent / name
        if not path.is_file() or sha256_lf(path) != expected:
            problems.append(f"Renderer changed since the receipt: {name}")
    for name, expected in receipt["outputs_sha256"].items():
        path = ROOT.parent / name
        if not path.is_file() or sha256_output(path) != expected:
            problems.append(f"Output does not match the receipt: {name}")
    for name in ("ragged-pool-clair.png", "ragged-pool-clair.svg",
                 "ragged-pool-clair-narrow.png", "ragged-pool-clair-narrow.svg"):
        if (ROOT / name).exists():
            problems.append(f"Figure not on Obscur: docs/{name}")
    if receipt["tokens"]["values"] != TOKENS:
        problems.append("Receipt tokens differ from the renderer")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--font-dir", type=Path, help="Folder with static Inter TTF files")
    parser.add_argument("--check", action="store_true", help="Compare with the committed files")
    parser.add_argument("--check-receipt", action="store_true",
                        help="Recompute the receipt hashes without rendering")
    args = parser.parse_args()
    if args.check_receipt:
        problems = check_receipt()
        if problems:
            parser.exit(1, "\n".join(problems) + "\n")
        print("Every hash in the rendering receipt matches.")
        return
    if not args.font_dir:
        parser.error("--font-dir is required to render")
    outputs = build(args.font_dir)
    if args.check:
        def same(path, data):
            if path.suffix == ".png":
                return path.read_bytes() == data
            return path.read_bytes().replace(b"\r\n", b"\n") == data
        stale = [name for name, data in outputs.items()
                 if not (ROOT / name).is_file() or not same(ROOT / name, data)]
        if stale:
            parser.exit(1, "Out of date: " + ", ".join(stale) + "\n")
        print(f"All {len(outputs)} diagram files match the renderer.")
        return
    for name, data in outputs.items():
        (ROOT / name).write_bytes(data)
    print(f"Rendered {len(outputs)} diagram files.")


if __name__ == "__main__":
    main()
