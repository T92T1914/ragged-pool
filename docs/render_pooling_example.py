"""Render a conceptual pooling diagram without running the operator or a study.

Run from any directory. The standard library produces the SVG files. Add
--font-dir pointing to official static Inter TTF files to also produce PNGs.
PNG rendering requires Pillow and fonttools. Font files are not distributed.
SVG visitors may use a sans serif fallback when Inter is not available.
Use --check to compare the selected outputs without changing them.
"""

from __future__ import annotations

import argparse
from html import escape
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOKEN_SOURCE = (
    "https://github.com/T92T1914/clair-obscur-themes/blob/"
    "7a57fe750ff50205a17e1d342106a0d3f2777159/tokens.json"
)
SOURCE_REVISION = "18c0d8f3f9750dd18e97940bde905bec08e6176e"
THEMES = {
    "clair": {
        "canvas": "#f8f7f3", "panel": "#fdfcf8", "text": "#242424",
        "muted": "#595854", "divider": "#b8b6ae", "accent": "#365f78",
        "on_accent": "#ffffff", "border": "#7f7d76",
    },
    "obscur": {
        "canvas": "#090909", "panel": "#151515", "text": "#f4f4f4",
        "muted": "#bcbcbc", "divider": "#494949", "accent": "#b3c8d8",
        "on_accent": "#15212c", "border": "#858585",
    },
}
TITLE = "Ragged Pooling for PyTorch"
DESCRIPTION = (
    "A conceptual example with groups of two items, three items and no items. "
    "Softmax turns each nonempty group's scores into weights. A weighted sum "
    "produces one vector per group. Empty groups produce a zero vector. "
    "Training sends gradients back to values and scores. CPU implementation. "
    "This illustration does not report a benchmark or model quality result."
)


def font_files(directory: Path):
    """Check real Inter faces and glyph coverage before rendering PNG text."""
    from fontTools.ttLib import TTFont

    faces = {}
    glyphs = set(range(32, 127))
    for name, weight in (("Regular", 400), ("SemiBold", 600), ("Bold", 700)):
        path = directory / f"Inter-{name}.ttf"
        with TTFont(path) as face:
            if (face["name"].getDebugName(6) != f"Inter-{name}"
                    or face["OS/2"].usWeightClass != weight
                    or face["OS/2"].fsSelection & 1
                    or not glyphs.issubset(face.getBestCmap())):
                raise ValueError(f"Incorrect Inter face or missing glyphs: {path.name}")
        faces[weight] = path
    return faces


class Canvas:
    """Share geometry and labels across vector and raster output."""

    def __init__(self, width, height, theme, faces=None):
        self.width, self.height, self.t = width, height, theme
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(TITLE)}</title>',
            f'<desc id="desc">{escape(DESCRIPTION)}</desc>',
            f'<!-- Family roles: {TOKEN_SOURCE}. Conceptual source: {SOURCE_REVISION}. -->',
        ]
        self.image, self.draw, self.faces = None, None, faces
        self.fonts = {}
        self.text_bounds = []
        if faces:
            from PIL import Image, ImageDraw
            self.image = Image.new("RGB", (width, height), theme["canvas"])
            self.draw = ImageDraw.Draw(self.image)
        self.rect(0, 0, width, height, "canvas", radius=0)

    def color(self, role):
        return self.t.get(role, role)

    def rect(self, x, y, w, h, fill, stroke=None, radius=16, line=2):
        f, s = self.color(fill), self.color(stroke) if stroke else "none"
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                        f'fill="{f}" stroke="{s}" stroke-width="{line}"/>')
        if self.draw:
            self.draw.rounded_rectangle((x, y, x+w, y+h), radius, f,
                                        None if s == "none" else s, width=line)

    def line(self, x1, y1, x2, y2, role="accent", width=4):
        color = self.color(role)
        self.svg.append(f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" '
                        f'stroke-width="{width}" fill="none" stroke-linecap="round"/>')
        if self.draw:
            self.draw.line((x1, y1, x2, y2), color, width)

    def arrow(self, x1, y1, x2, y2, role="accent", width=4):
        self.line(x1, y1, x2, y2, role, width)
        direction = 1 if x2 > x1 else -1
        self.line(x2 - direction*12, y2-9, x2, y2, role, width)
        self.line(x2 - direction*12, y2+9, x2, y2, role, width)

    def text(self, x, y, value, size=32, weight=400, role="text", anchor="start"):
        color = self.color(role)
        self.svg.append(
            f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')
        if self.draw:
            from PIL import ImageFont
            key = (size, weight)
            if key not in self.fonts:
                self.fonts[key] = ImageFont.truetype(str(self.faces[weight]), size)
            font = self.fonts[key]
            pil_anchor = {"start": "ls", "middle": "ms", "end": "rs"}[anchor]
            box = self.draw.textbbox((x, y), value, font=font, anchor=pil_anchor)
            self.text_bounds.append((value, box))
            if box[0] < 0 or box[1] < 0 or box[2] > self.width or box[3] > self.height:
                raise ValueError(f"Text exceeds the canvas: {value}")
            self.draw.text((x, y), value, color, font=font, anchor=pil_anchor)

    def vector(self, x, y, zero=False, scale=1):
        w, h = round(116*scale), round(44*scale)
        self.rect(x, y, w, h, "panel", "accent", radius=round(9*scale))
        for i in range(3):
            cx = x + (20 + i*38)*scale
            if zero:
                self.text(cx, y+30*scale, "0", round(26*scale), anchor="middle", role="accent")
            else:
                self.line(cx-4*scale, y+22*scale, cx+4*scale, y+22*scale,
                          "accent", round(7*scale))

    def weight(self, x, y, width=98):
        self.rect(x, y, width, 22, "panel", "accent", radius=8)
        self.line(x+12, y+11, x+width-12, y+11, "accent", 5)

    def finish(self):
        svg = ("\n".join(self.svg + ["</svg>"]) + "\n").encode("utf-8")
        png = None
        if self.image:
            stream = BytesIO()
            self.image.save(stream, "PNG", optimize=False, compress_level=9)
            png = stream.getvalue()
        return svg, png


def render(edition, narrow=False, faces=None):
    c = Canvas(900 if narrow else 1600, 1590 if narrow else 1120, THEMES[edition], faces)
    if narrow:
        c.text(54, 93, "Ragged Pooling", 58, 600)
        c.text(54, 155, "for PyTorch", 58, 600)
        c.text(54, 217, "Different group sizes.", 35, role="muted")
        c.text(54, 260, "One summary vector per group.", 35, role="muted")
        c.rect(40, 301, 820, 946, "panel", "divider", 22)
        c.text(72, 355, "Values and scores", 34, 600)
        c.text(72, 398, "Softmax weights feed a weighted sum.", 29, role="muted")
        for row, (label, count) in enumerate((("Group A", 2), ("Group B", 3), ("Empty group", 0))):
            y = 448 + row*240
            c.text(72, y, label, 31, 600)
            if count:
                for item in range(count):
                    c.vector(72 + item*143, y+23)
                    c.weight(81 + item*143, y+80)
                c.text(72, y+143, "Items and their weights", 27, role="muted")
            else:
                c.text(72, y+73, "No items", 32, role="muted")
            c.arrow(520, y+66, 636, y+66)
            c.vector(686, y+44, zero=not count)
            c.text(744, y+143, "Zero vector" if not count else "One vector", 27,
                   anchor="middle", role="muted")
            if row < 2:
                c.line(72, y+187, 828, y+187, "divider", 2)
        c.arrow(790, 1201, 110, 1201)
        c.text(450, 1167, "Gradients reach values and scores", 31, 600,
               anchor="middle", role="accent")
        c.text(54, 1324, "CPU computation in C++.", 34, 600)
        c.text(54, 1375, "Python integration with PyTorch.", 32)
        c.text(54, 1424, "Packed and padded inputs.", 32)
        c.text(54, 1473, "First order gradients.", 32)
        c.text(54, 1542, "Conceptual example. Not a benchmark.", 29, role="muted")
    else:
        c.text(70, 116, TITLE, 66, 600)
        c.text(70, 184, "Different group sizes. One summary vector per group.", 36, role="muted")
        c.rect(56, 238, 1488, 585, "panel", "divider", 22)
        c.text(94, 300, "Values and scores", 32, 600)
        c.text(700, 300, "Softmax weights", 32, 600)
        c.text(1246, 300, "Weighted sum", 32, 600)
        for row, (label, count) in enumerate((("Group A", 2), ("Group B", 3), ("Empty group", 0))):
            y = 342 + row*146
            c.text(94, y+36, label, 29, 600)
            if count:
                for item in range(count):
                    c.vector(282 + item*131, y+6)
                    c.weight(710 + item*137, y+17, width=108)
            else:
                c.text(340, y+36, "No items", 30, role="muted")
                c.text(710, y+36, "No weights", 30, role="muted")
            c.arrow(1160, y+28, 1225, y+28)
            c.vector(1314, y+6, zero=not count)
            if row < 2:
                c.line(94, y+90, 1505, y+90, "divider", 2)
        c.text(800, 868, "Training sends gradients back to values and scores", 34,
               600, "accent", "middle")
        c.arrow(1395, 910, 220, 910)
        c.text(70, 969, "CPU computation in C++. Python integration with PyTorch.", 34, 600)
        c.text(70, 1024, "Packed and padded inputs. First order gradients.", 33)
        c.text(70, 1080, "Conceptual example. Not a benchmark.", 29, role="muted")
    return c.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, help="Directory with static Inter TTF files for PNGs")
    parser.add_argument("--check", action="store_true", help="Compare selected files without changing them")
    args = parser.parse_args()
    faces = font_files(args.font_dir) if args.font_dir else None
    outputs = {}
    for edition in THEMES:
        for narrow in (False, True):
            name = f"ragged-pool-{edition}{'-narrow' if narrow else ''}"
            svg, png = render(edition, narrow, faces)
            outputs[ROOT / f"{name}.svg"] = svg
            if png is not None:
                outputs[ROOT / f"{name}.png"] = png
    stale = [path.name for path, data in outputs.items()
             if not path.is_file() or path.read_bytes() != data]
    if args.check:
        if stale:
            parser.exit(1, "Out of date: " + ", ".join(stale) + "\n")
        print(f"All {len(outputs)} selected diagram files match the renderer.")
    else:
        for path, data in outputs.items():
            path.write_bytes(data)
        print(f"Rendered {len(outputs)} conceptual diagram files.")


if __name__ == "__main__":
    main()
