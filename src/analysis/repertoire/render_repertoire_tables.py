"""Renders the repertoire results tables to a PDF and a PNG of Table 1.

Reads results/repertoire/repertoire_tables.md (written by
build_repertoire_tables.py), converts it to HTML, and uses headless Chrome to
print the whole document to a PDF and to screenshot Table 1 (with its title and
notes, without the long explanations) as a PNG.

The page is built the way the VS Code extension Markdown Preview Enhanced builds
its export: the extension's "github-light" theme (github_light_theme.css, copied
from the extension) inside a div.markdown-preview, printed by Chrome on US
Letter. The PDF therefore looks the same as the extension's "Chrome (Puppeteer)"
PDF export, and rerunning this script after build_repertoire_tables.py keeps it
in step with the tables.

The converter handles only the markdown that the table builder writes: headings,
paragraphs, tables, bullet lists, **bold**, *italic* and `code`.

Inputs:
    results/repertoire/repertoire_tables.md
    src/analysis/repertoire/github_light_theme.css

Outputs:
    results/repertoire/repertoire_tables.pdf
    results/repertoire/repertoire_table_1.png

Usage (from the project root):
    python src/analysis/repertoire/build_repertoire_tables.py
    python src/analysis/repertoire/render_repertoire_tables.py
"""

import html
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
RESULTS_DIR = PROJECT_ROOT / "results" / "repertoire"
MARKDOWN_PATH = RESULTS_DIR / "repertoire_tables.md"
PDF_PATH = RESULTS_DIR / "repertoire_tables.pdf"
PNG_PATH = RESULTS_DIR / "repertoire_table_1.png"
THEME_PATH = Path(__file__).with_name("github_light_theme.css")

CHROME_PATHS = [
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
]

# the base rules the extension adds around its theme in an HTML export
EXPORT_CSS = """
.markdown-preview{width:100%;height:100%;box-sizing:border-box}
.markdown-preview ul{list-style:disc}
.markdown-preview ul ul{list-style:circle}
.markdown-preview ul ul ul{list-style:square}
html body[for=html-export]:not([data-presentation-mode]){position:relative;
width:100%;height:100%;top:0;left:0;margin:0;padding:0;overflow:auto}
html body[for=html-export]:not([data-presentation-mode]) .markdown-preview{
position:relative;top:0;min-height:100vh}
html body[for=html-export]:not([data-presentation-mode]):not([html-show-sidebar-toc])
.markdown-preview{left:50%;transform:translateX(-50%)}
@media screen and (min-width:914px){
html body[for=html-export]:not([data-presentation-mode]) .markdown-preview{
padding:2em calc(50% - 457px + 2em)}}
"""

# page setup for the PDF: US Letter, with the 8px side inset the extension's
# export has (measured against its PDF)
PDF_CSS = """
@page { size: Letter; }
html body[for=html-export] .markdown-preview { padding: 0 8px; }
"""

# the PNG uses the full width of the window instead of the extension's 914px
# column, so the table is not squeezed
PNG_CSS = """
html body[for=html-export] .markdown-preview{padding:24px !important;left:0 !important;
transform:none !important}
"""


def inline(text: str) -> str:
    """Escapes HTML and converts `code`, **bold** and *italic* to tags.

    Args:
        text (str): One line or paragraph of markdown.

    Returns:
        str: The HTML.
    """
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)


def parse_blocks(markdown: str) -> list[dict]:
    """Splits the markdown into headings, paragraphs, tables and lists.

    Args:
        markdown (str): The contents of repertoire_tables.md.

    Returns:
        list[dict]: Blocks with a "type" (h1, h2, p, table or list) and the
            matching "text", "rows" or "items".
    """
    lines = markdown.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("## "):
            blocks.append({"type": "h2", "text": line[3:]})
            i += 1
        elif line.startswith("# "):
            blocks.append({"type": "h1", "text": line[2:]})
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows.pop(1)  # the |---|---| separator row
            blocks.append({"type": "table", "rows": rows})
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            blocks.append({"type": "list", "items": items})
        else:
            paragraph = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].startswith(("|", "- ", "#"))
            ):
                paragraph.append(lines[i])
                i += 1
            blocks.append({"type": "p", "text": " ".join(paragraph)})
    return blocks


def block_html(block: dict) -> str:
    """Converts one block to HTML, in the structure markdown-it produces."""
    kind = block["type"]
    if kind in ("h1", "h2"):
        return f"<{kind}>{inline(block['text'])}</{kind}>"
    if kind == "p":
        return f"<p>{inline(block['text'])}</p>"
    if kind == "list":
        items = "\n".join(f"<li>{inline(i)}</li>" for i in block["items"])
        return f"<ul>\n{items}\n</ul>"
    head, *body = block["rows"]
    out = ["<table>", "<thead>", "<tr>"]
    out += [f"<th>{inline(c)}</th>" for c in head]
    out += ["</tr>", "</thead>", "<tbody>"]
    for row in body:
        out.append("<tr>")
        out += [f"<td>{inline(c)}</td>" for c in row]
        out.append("</tr>")
    return "\n".join(out + ["</tbody>", "</table>"])


def page_html(blocks: list[dict], extra_css: str) -> str:
    """Builds a whole HTML page from blocks, in the extension's export structure.

    Args:
        blocks (list[dict]): Output of parse_blocks, or a selection of it.
        extra_css (str): Further CSS, e.g. the page setup for the PDF.

    Returns:
        str: The HTML page.
    """
    css = THEME_PATH.read_text(encoding="utf-8") + EXPORT_CSS + extra_css
    body = "\n".join(block_html(b) for b in blocks)
    return (
        '<!doctype html><html lang="en-GB"><head><meta charset="utf-8">'
        f"<title>repertoire_tables</title><style>{css}</style></head>"
        f'<body for="html-export"><div class="markdown-preview">{body}</div>'
        "</body></html>"
    )


def table_1_blocks(blocks: list[dict]) -> list[dict]:
    """Selects Table 1's heading, table and notes, without the long text.

    Keeps the heading, the table, the dagger line and the "Other notes" list,
    so the image can be read on its own; leaves out the explainer and the
    definition of the effective number of blocks.

    Args:
        blocks (list[dict]): Output of parse_blocks.

    Returns:
        list[dict]: The selected blocks.
    """
    start = next(
        i
        for i, b in enumerate(blocks)
        if b["type"] == "h2" and b["text"].startswith("Table 1")
    )
    end = next(
        (i for i, b in enumerate(blocks) if i > start and b["type"] == "h2"),
        len(blocks),
    )
    keep = []
    for b in blocks[start:end]:
        if b["type"] in ("h2", "table", "list"):
            keep.append(b)
        elif b["type"] == "p" and (
            b["text"].startswith("\u2020") or b["text"] == "Other notes:"
        ):
            keep.append(b)
    return keep


def find_chrome() -> Path:
    """Returns the path to Chrome.

    Raises:
        FileNotFoundError: If Chrome is not in a standard location.
    """
    for path in CHROME_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Chrome not found; add its path to CHROME_PATHS.")


def run_chrome(chrome: Path, profile: Path, flags: list[str], page: Path) -> None:
    """Runs headless Chrome on a local HTML file.

    Args:
        chrome (Path): The Chrome executable.
        profile (Path): A throwaway profile folder, so a Chrome window that is
            already open does not interfere.
        flags (list[str]): Output flags, e.g. --print-to-pdf=...
        page (Path): The HTML file to load.
    """
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        *flags,
        page.as_uri(),
    ]
    subprocess.run(command, check=True, timeout=120, capture_output=True)


def crop_whitespace(source: Path, target: Path, margin: int = 24) -> None:
    """Crops a screenshot to its content, with a small white margin.

    Args:
        source (Path): The full screenshot.
        target (Path): Where to save the cropped image.
        margin (int): White margin to keep, in pixels.
    """
    image = Image.open(source).convert("RGB")
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    left, top, right, bottom = box
    image.crop(
        (
            max(left - margin, 0),
            max(top - margin, 0),
            min(right + margin, image.width),
            min(bottom + margin, image.height),
        )
    ).save(target)


def main() -> None:
    """Renders the PDF of the whole file and the PNG of Table 1."""
    blocks = parse_blocks(MARKDOWN_PATH.read_text(encoding="utf-8"))
    chrome = find_chrome()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        tmp = Path(folder)
        (tmp / "all.html").write_text(page_html(blocks, PDF_CSS), encoding="utf-8")
        (tmp / "table1.html").write_text(
            page_html(table_1_blocks(blocks), PNG_CSS), encoding="utf-8"
        )
        run_chrome(
            chrome,
            tmp / "profile_pdf",
            [f"--print-to-pdf={PDF_PATH}", "--no-pdf-header-footer"],
            tmp / "all.html",
        )
        run_chrome(
            chrome,
            tmp / "profile_png",
            [
                f"--screenshot={tmp / 'full.png'}",
                "--window-size=1300,1800",
                "--force-device-scale-factor=2",
            ],
            tmp / "table1.html",
        )
        crop_whitespace(tmp / "full.png", PNG_PATH)
    print(f"Saved {PDF_PATH.name} and {PNG_PATH.name}")


if __name__ == "__main__":
    main()
