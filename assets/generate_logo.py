"""Generate a PNG logo for speqtro using the same pixel font and benzene ring from the UI.

Usage:
    python generate_logo.py          # -> logo.png (dark background)
    python generate_logo.py --light  # -> logo.png (transparent background)

Requires: pip install Pillow
"""

import argparse
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required: pip install Pillow")
    raise SystemExit(1)

# --- Pixel font (from ui/terminal.py) ---
PIXEL_FONT = {
    "S": [
        " ███ ",
        "█   █",
        "█    ",
        " ███ ",
        "    █",
        "█   █",
        " ███ ",
    ],
    "P": [
        "████ ",
        "█   █",
        "█   █",
        "████ ",
        "█    ",
        "█    ",
        "█    ",
    ],
    "E": [
        "█████",
        "█    ",
        "█    ",
        "████ ",
        "█    ",
        "█    ",
        "█████",
    ],
    "Q": [
        " ███ ",
        "█   █",
        "█   █",
        "█   █",
        "█ ██ ",
        "█  ██",
        " ████",
    ],
    "T": [
        "█████",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
    ],
    "R": [
        "████ ",
        "█   █",
        "█   █",
        "████ ",
        "█ █  ",
        "█  █ ",
        "█   █",
    ],
    "O": [
        " ███ ",
        "█   █",
        "█   █",
        "█   █",
        "█   █",
        "█   █",
        " ███ ",
    ],
}


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def gradient_color(row: int, total_rows: int) -> tuple:
    """Lime (#c8ff3a) -> Cyan (#00e5ff) vertical gradient."""
    if total_rows <= 1:
        return (0xC8, 0xFF, 0x3A)
    t = row / (total_rows - 1)
    return lerp_color((0xC8, 0xFF, 0x3A), (0x00, 0xE5, 0xFF), t)


def pulse_color(phase: float) -> tuple:
    """Green (#50fa7b) -> Cyan (#00e5ff) pulse for the benzene ring."""
    t = (math.sin(phase) + 1) / 2
    return lerp_color((0x50, 0xFA, 0x7B), (0x00, 0xE5, 0xFF), t)


def draw_benzene(draw: ImageDraw.Draw, cx: int, cy: int, radius: int,
                 highlight_idx: int = 0, highlight_color: tuple = (0x50, 0xFA, 0x7B)):
    """Draw a hexagonal benzene ring with one highlighted vertex."""
    dim_color = (0x2A, 0x2A, 0x5A)
    bond_color = (0x1E, 0x1E, 0x3E)
    node_r = max(3, radius // 8)

    # Compute vertices (flat-top hexagon, clockwise from top-right)
    verts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)  # start top-right
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        verts.append((x, y))

    # Draw bonds
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        draw.line([(x1, y1), (x2, y2)], fill=bond_color, width=max(2, radius // 12))

    # Inner ring (alternating double bonds) — 3 shorter bonds
    inner_r = radius * 0.65
    inner_verts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        x = cx + inner_r * math.cos(angle)
        y = cy + inner_r * math.sin(angle)
        inner_verts.append((x, y))
    for i in [0, 2, 4]:
        x1, y1 = inner_verts[i]
        x2, y2 = inner_verts[(i + 1) % 6]
        draw.line([(x1, y1), (x2, y2)], fill=bond_color, width=max(1, radius // 16))

    # Draw atom nodes
    for i, (x, y) in enumerate(verts):
        color = highlight_color if i == highlight_idx else dim_color
        r = node_r + 2 if i == highlight_idx else node_r
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def generate_logo(dark: bool = True, output: str = "logo.png"):
    pixel_size = 8       # each pixel-font cell = 8×8 px
    letter_gap = 16      # px between letters
    word = "SPEQTRO"
    height = 7           # rows in pixel font

    # Measure total width
    letter_widths = [len(PIXEL_FONT[ch][0]) for ch in word]
    total_cells = sum(letter_widths)
    text_w = total_cells * pixel_size + (len(word) - 1) * letter_gap
    text_h = height * pixel_size

    # Image dimensions
    benzene_radius = 40
    padding = 40
    ring_gap = 30

    img_w = padding + text_w + ring_gap + benzene_radius * 2 + padding
    img_h = max(text_h, benzene_radius * 2) + padding * 2

    bg = (0x0C, 0x0C, 0x1A) if dark else (0, 0, 0, 0)
    mode = "RGB" if dark else "RGBA"
    img = Image.new(mode, (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # Draw pixel-art SPEQTRO
    text_y0 = (img_h - text_h) // 2
    cursor_x = padding
    for ch in word:
        glyph = PIXEL_FONT[ch]
        for row_i, row_str in enumerate(glyph):
            color = gradient_color(row_i, height)
            for col_i, cell in enumerate(row_str):
                if cell == "\u2588":  # █
                    x = cursor_x + col_i * pixel_size
                    y = text_y0 + row_i * pixel_size
                    draw.rectangle([x, y, x + pixel_size - 1, y + pixel_size - 1], fill=color)
        cursor_x += len(glyph[0]) * pixel_size + letter_gap

    # Accent line under text
    line_y = text_y0 + text_h + 6
    line_x0 = padding + 4
    line_x1 = cursor_x - letter_gap - 4
    draw.line([(line_x0, line_y), (line_x1, line_y)], fill=(0x00, 0xC8, 0xE8), width=2)

    # Draw benzene ring
    ring_cx = cursor_x + ring_gap + benzene_radius
    ring_cy = img_h // 2
    hl_color = pulse_color(0.0)  # static snapshot at green phase
    draw_benzene(draw, ring_cx, ring_cy, benzene_radius,
                 highlight_idx=0, highlight_color=hl_color)

    # Subtitle
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    sub = "spectroscopy reasoning agent"
    sub_bbox = draw.textbbox((0, 0), sub, font=font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    sub_x = padding + (text_w - sub_w) // 2
    sub_y = line_y + 8
    draw.text((sub_x, sub_y), sub, fill=(0x66, 0x88, 0x99), font=font)

    img.save(output)
    print(f"Saved {output} ({img_w}×{img_h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speqtro logo")
    parser.add_argument("--light", action="store_true", help="Transparent background")
    parser.add_argument("-o", "--output", default="logo.png", help="Output filename")
    args = parser.parse_args()
    generate_logo(dark=not args.light, output=args.output)
