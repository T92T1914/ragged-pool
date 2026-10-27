"""Build and validate only the static explanation; never import the native package."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "ddf8002cfd351723ff7a9e6c7471a966b8fe33db"
THEME = "7a57fe750ff50205a17e1d342106a0d3f2777159"
FILES = {
    "site/index.html": "index.html",
    "site/style.css": "style.css",
    "site/appearance.js": "appearance.js",
    "docs/ragged-pool-obscur.png": "ragged-pool-obscur.png",
    "docs/ragged-pool-obscur-narrow.png": "ragged-pool-obscur-narrow.png",
}
ASSETS = {
    "ragged-pool-obscur.png": "108403fa1d1eef675d70ceb583d33c55d52fdbb26e787da1b1ba3e9038e2f229",
    "ragged-pool-obscur-narrow.png": "9cf0b7b0f0ba676fc2331f0d4af531d6d9b22038be7b19550cff7e2a6eb941b1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []
        self.images = []
        self.html = {}
        self.scripts = []

    def handle_starttag(self, tag, items):
        attrs = dict(items)
        if tag == "html":
            self.html = attrs
        if attrs.get("id"):
            require(attrs["id"] not in self.ids, "Duplicate HTML id")
            self.ids.add(attrs["id"])
        if tag == "img":
            self.images.append(attrs)
            require(bool(attrs.get("alt", "").strip()), "Missing diagram alternative")
        if tag == "script":
            self.scripts.append(attrs.get("src", ""))
        for key in ("src", "href"):
            if attrs.get(key):
                self.references.append(attrs[key])
        if attrs.get("srcset"):
            # These maintained sources each name one width-specific image.
            require("," not in attrs["srcset"], "Use a separate source for each edition")
            self.references.append(attrs["srcset"])


def validate_page(text, names):
    page = Page()
    page.feed(text)
    require(page.html.get("data-appearance") == "obscur", "Missing no-script Obscur default")
    require(page.html.get("lang") == "en", "Missing page language")
    require(page.images, "Missing explanatory image")
    require(page.scripts == ["appearance.js"], "Unexpected page script")
    require(text.index('src="appearance.js"') < text.index('href="style.css"'),
            "Resolve saved appearance before styles paint")
    for reference in page.references:
        parsed = urlsplit(reference)
        if parsed.scheme:
            require(parsed.scheme == "https", "Only explicit HTTPS external routes are admitted")
            continue
        require(not parsed.netloc, "Protocol-relative resource")
        name = unquote(parsed.path)
        require(not name or name in names, f"Unpublished local resource: {name}")
        if parsed.fragment and not name:
            require(parsed.fragment in page.ids, f"Missing page anchor: {parsed.fragment}")
    return page


def validate_sources(root=ROOT):
    files = {target: (root / source).read_bytes() for source, target in FILES.items()}
    text = files["index.html"].decode("utf-8")
    page = validate_page(text, set(FILES.values()) | {"site-record.json"})
    for name, expected in ASSETS.items():
        require(hashlib.sha256(files[name]).hexdigest() == expected,
                f"Retained diagram changed: {name}")
    for reference in page.references:
        marker = f"https://github.com/T92T1914/ragged-pool/blob/{RELEASE}/"
        if reference.startswith(marker):
            local = root / urlsplit(reference[len(marker):]).path
            require(local.is_file(), f"Missing referenced project source: {local.name}")
    style = files["style.css"].decode("utf-8")
    require(THEME in style, "Missing pinned family source")
    require("filter:" not in style, "Original media must not be recolored")
    require("url(" not in style, "Do not add a remote font or media request")
    require(style.count("@font-face") == 6, "Keep the local six-face font route")
    require('prefers-reduced-motion' in style and 'forced-colors' in style
            and '@media print' in style, "Keep user media preferences")
    return files


def source_identity(root):
    # Read-only Git commands, no shell, hook or package import.
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    require(bool(re.fullmatch("[0-9a-f]{40}", revision)), "Invalid source revision")
    return revision, bool(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-built", action="store_true",
                        help="Compare an existing build with the current static sources")
    args = parser.parse_args()
    files = validate_sources()
    revision, dirty = source_identity(ROOT)
    record = {
        "schema_version": 1,
        "presentation_revision": revision,
        "presentation_worktree_dirty": dirty,
        "runtime_release_revision": RELEASE,
        "theme_revision": THEME,
        "figure_kind": "Reused conceptual CPU diagram, not a new execution",
        "evaluation_rerun": False,
        "files_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in sorted(files.items())},
    }
    output = ROOT / "_site"
    expected = set(files) | {"site-record.json"}
    if output.exists():
        require(output.is_dir() and not output.is_symlink(), "Output must be an ordinary directory")
        require({item.name for item in output.iterdir()} <= expected,
                "Unexpected existing output; preserve it rather than delete it")
    if args.check_built:
        require(output.is_dir(), "Build the site before comparing output")
        require({item.name for item in output.iterdir()} == expected, "Incomplete static output")
        for name, data in files.items():
            require((output / name).read_bytes() == data, f"Stale static output: {name}")
        saved = json.loads((output / "site-record.json").read_text(encoding="utf-8"))
        require(saved == record, "Stale source/asset record")
    else:
        output.mkdir(exist_ok=True)
        for source, target in FILES.items():
            require(not (output / target).is_symlink(), "Refuse a symlink output")
            shutil.copyfile(ROOT / source, output / target)
        require(not (output / "site-record.json").is_symlink(), "Refuse a symlink record")
        (output / "site-record.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n",
        )
    print(json.dumps({"static_files": len(expected), "source_checks": "passed",
                      "check_built": args.check_built, "evaluation_rerun": False}))


if __name__ == "__main__":
    main()

