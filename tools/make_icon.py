#!/usr/bin/env python3
"""Draw the app icon: a smooth voice wave that turns into autotune "steps".

Writes assets/icon.png (1024px, used for macOS .icns and the window icon)
and assets/icon.ico (Windows, multi-size).
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024          # final size
SS = 4            # supersampling for smooth edges
N = S * SS
OUT = os.path.join(os.path.dirname(__file__), "..", "assets")


def squircle_mask(size, margin, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([margin, margin, size - margin, size - margin],
                                        radius=radius, fill=255)
    return m


def main():
    # diagonal gradient: Discord blurple -> fuchsia
    y, x = np.mgrid[0:N, 0:N] / (N - 1)
    t = np.clip((x * 0.55 + y * 0.45), 0, 1)[..., None]
    c0 = np.array([88, 101, 242])    # #5865F2
    c1 = np.array([235, 69, 158])    # #EB459E
    bg = (c0 * (1 - t) + c1 * t).astype(np.uint8)
    img = Image.fromarray(np.dstack([bg, np.full((N, N), 255, np.uint8)]), "RGBA")

    margin = int(N * 0.06)
    mask = squircle_mask(N, margin, int(N * 0.22))

    # soft light from the top
    glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([N * 0.05, -N * 0.55, N * 0.95, N * 0.4],
                                 fill=(255, 255, 255, 45))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(N * 0.08)))

    x0, x1 = margin + N * 0.12, N - margin - N * 0.12
    cy, amp = N * 0.5, N * 0.2

    def wave_at(xv):
        p = (xv - x0) / (x1 - x0)
        return cy - amp * np.sin(p * 2 * np.pi * 2.0)

    def layer(draw_fn, blur=0):
        lay = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(lay))
        return lay.filter(ImageFilter.GaussianBlur(blur)) if blur else lay

    # left half: the voice as it is (smooth); right half: snapped to notes (steps)
    xm = (x0 + x1) / 2
    xs = np.linspace(x0, xm, 2500)
    smooth = list(zip(xs, wave_at(xs)))
    pts = [smooth[-1]]
    steps, levels = 8, 3
    step_h = amp / levels
    edges = np.linspace(xm, x1, steps + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        v = wave_at((a + b) / 2)
        v = cy + round((v - cy) / step_h) * step_h
        pts += [(a, v), (b, v)]
    w = int(N * 0.052)
    r = w / 2

    def path(d, fill, dy=0):
        # smooth part: stamped circles (a thick PIL polyline with many points frays)
        for px, py in smooth:
            d.ellipse([px - r, py + dy - r, px + r, py + dy + r], fill=fill)
        q = [(px, py + dy) for px, py in pts]
        d.line(q, fill=fill, width=w, joint="curve")
        px, py = q[-1]  # round cap
        d.ellipse([px - r, py - r, px + r, py + r], fill=fill)

    img = Image.alpha_composite(img, layer(lambda d: path(d, (40, 20, 90, 120), N * 0.014),
                                           blur=N * 0.014))
    img = Image.alpha_composite(img, layer(lambda d: path(d, (255, 255, 255, 255))))

    img.putalpha(mask)
    img = img.resize((S, S), Image.LANCZOS)
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, "icon.png"))
    img.save(os.path.join(OUT, "icon.ico"),
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("assets/icon.png, assets/icon.ico")


if __name__ == "__main__":
    main()
