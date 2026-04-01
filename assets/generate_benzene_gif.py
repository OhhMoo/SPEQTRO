"""Generate an animated GIF of the speqtro thinking benzene ring.

Reproduces the rotating benzene ring from ui/status.py:
- 6 atom positions on a hexagon
- One highlighted atom (neon green/cyan) orbits clockwise
- Green-cyan color pulse on the active atom
- Alternating double bonds inside the ring
- Dark background matching the terminal aesthetic

Usage:
    python generate_benzene_gif.py              # -> benzene.gif
    python generate_benzene_gif.py --with-text  # -> includes "Analyzing spectra..." text
    python generate_benzene_gif.py --size 300   # -> larger ring

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


# ---------------------------------------------------------------------------
# Thinking words (from ui/status.py)
# ---------------------------------------------------------------------------
THINKING_WORDS = [
    "Analyzing spectra",
    "Scanning NMR peaks",
    "Mapping chemical shifts",
    "Identifying functional groups",
    "Reviewing MS fragmentation",
    "Calculating exact mass",
    "Matching IR absorptions",
    "Assigning carbon skeleton",
    "Evaluating coupling constants",
    "Surveying chemical space",
]


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def gradient_color(elapsed_s: float) -> tuple:
    """Green (#50fa7b) -> Cyan (#00e5ff) pulse, matching status.py."""
    cycle = 2.25
    t = (math.sin((elapsed_s % cycle) * (2 * math.pi / cycle)) + 1) / 2
    return lerp_color((0x50, 0xFA, 0x7B), (0x00, 0xE5, 0xFF), t)


def draw_benzene_frame(
    img_w: int,
    img_h: int,
    radius: int,
    active_idx: int,
    elapsed_s: float,
    with_text: bool = False,
    word: str = "",
    time_str: str = "",
) -> Image.Image:
    """Render one frame of the benzene animation."""
    bg = (0x0C, 0x0C, 0x1A)
    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # Ring center
    if with_text:
        cx = img_w // 4
    else:
        cx = img_w // 2
    cy = img_h // 2

    dim_color = (0x2A, 0x2A, 0x5A)
    bond_color = (0x1E, 0x1E, 0x3E)
    bond_color_bright = (0x30, 0x30, 0x60)
    hl_color = gradient_color(elapsed_s)

    node_r = max(4, radius // 6)
    bond_w = max(2, radius // 14)

    # Compute vertices (flat-top hexagon, clockwise from top-right to match status.py)
    # status.py order: top-right, right, bottom-right, bottom-left, left, top-left
    verts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        verts.append((x, y))

    # Draw outer bonds
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        draw.line([(x1, y1), (x2, y2)], fill=bond_color_bright, width=bond_w)

    # Inner ring — alternating double bonds (3 bonds)
    inner_r = radius * 0.62
    inner_verts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        x = cx + inner_r * math.cos(angle)
        y = cy + inner_r * math.sin(angle)
        inner_verts.append((x, y))
    for i in [0, 2, 4]:
        x1, y1 = inner_verts[i]
        x2, y2 = inner_verts[(i + 1) % 6]
        draw.line([(x1, y1), (x2, y2)], fill=bond_color, width=max(1, bond_w - 1))

    # Draw atom nodes
    for i, (x, y) in enumerate(verts):
        if i == active_idx:
            # Highlighted atom — larger, with glow
            glow_r = node_r + 6
            for gr in range(glow_r, node_r, -1):
                alpha_t = (gr - node_r) / (glow_r - node_r)
                glow_c = lerp_color(hl_color, bg, alpha_t * 0.8)
                draw.ellipse([x - gr, y - gr, x + gr, y + gr], fill=glow_c)
            draw.ellipse([x - node_r - 2, y - node_r - 2, x + node_r + 2, y + node_r + 2],
                         fill=hl_color)
        else:
            draw.ellipse([x - node_r, y - node_r, x + node_r, y + node_r], fill=dim_color)

    # Optional text
    if with_text and word:
        try:
            font = ImageFont.truetype("arial.ttf", max(14, radius // 4))
            font_small = ImageFont.truetype("arial.ttf", max(11, radius // 6))
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font

        text_x = cx + radius + max(20, radius // 2)
        text_y = cy - radius // 3

        # Word with ellipsis
        draw.text((text_x, text_y), f"{word}...", fill=(0x00, 0xE5, 0xFF), font=font)

        # Timer
        draw.text((text_x, text_y + max(22, radius // 3) + 8),
                  f"({time_str})", fill=(0x55, 0x55, 0x66), font=font_small)

    return img


def generate_gif(
    output: str = "benzene.gif",
    size: int = 200,
    with_text: bool = False,
    duration_s: float = 4.5,
    fps: int = 16,
):
    """Generate the animated GIF."""
    radius = size // 3

    if with_text:
        img_w = size * 3
    else:
        img_w = size
    img_h = size

    total_frames = int(duration_s * fps)
    frame_duration_ms = int(1000 / fps)

    frames = []
    for fi in range(total_frames):
        elapsed = fi / fps

        # One full rotation every 1.5s (matching status.py)
        active_idx = int(elapsed / 0.25) % 6

        # Rotate through words every 3s
        word = THINKING_WORDS[int(elapsed / 3) % len(THINKING_WORDS)]

        # Timer display
        time_str = f"{elapsed:.0f}s"

        frame = draw_benzene_frame(
            img_w, img_h, radius, active_idx, elapsed,
            with_text=with_text, word=word, time_str=time_str,
        )
        frames.append(frame)

    # Save as looping GIF
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,  # infinite loop
        optimize=True,
    )
    print(f"Saved {output} ({img_w}x{img_h}, {total_frames} frames, {duration_s}s loop)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speqtro benzene thinking GIF")
    parser.add_argument("-o", "--output", default="benzene.gif", help="Output filename")
    parser.add_argument("--size", type=int, default=200, help="Image height in px (default 200)")
    parser.add_argument("--with-text", action="store_true",
                        help="Include thinking words alongside the ring")
    parser.add_argument("--duration", type=float, default=4.5,
                        help="Loop duration in seconds (default 4.5)")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second (default 16)")
    args = parser.parse_args()

    generate_gif(
        output=args.output,
        size=args.size,
        with_text=args.with_text,
        duration_s=args.duration,
        fps=args.fps,
    )
