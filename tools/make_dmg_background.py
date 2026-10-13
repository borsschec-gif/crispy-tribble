#!/usr/bin/env python3
"""Draw the macOS installer (.dmg) window background.

Writes assets/dmg_background.png (660x440) and assets/dmg_background@2x.png.
The app icon and the Applications folder are placed on top by dmgbuild
(see tools/dmg_settings.py); this image only adds the dark backdrop and the arrow.
"""
import os

from PIL import Image, ImageDraw

W, H = 660, 440
APP_X, APPS_X, ICON_Y = 180, 480, 190  # keep in sync with dmg_settings.py
OUT = os.path.join(os.path.dirname(__file__), "..", "assets")


def draw(scale):
    w, h = W * scale, H * scale
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    # Discord-dark vertical gradient
    top, bottom = (54, 57, 63), (35, 36, 40)
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))

    # arrow from the app to Applications: blurple -> fuchsia dashes and a head
    s = scale
    y = ICON_Y * s
    x0, x1 = (APP_X + 90) * s, (APPS_X - 90) * s
    head = 16 * s
    n = 7
    for i in range(n):
        a = x0 + (x1 - head - x0) * i / n
        b = a + (x1 - head - x0) / n * 0.55
        t = i / (n - 1)
        col = tuple(int(c0 + (c1 - c0) * t) for c0, c1 in zip((88, 101, 242), (235, 69, 158)))
        d.rounded_rectangle([a, y - 3 * s, b, y + 3 * s], radius=3 * s, fill=col)
    d.polygon([(x1 - head, y - 12 * s), (x1, y), (x1 - head, y + 12 * s)], fill=(235, 69, 158))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    draw(1).save(os.path.join(OUT, "dmg_background.png"))
    draw(2).save(os.path.join(OUT, "dmg_background@2x.png"))
    print("assets/dmg_background.png, assets/dmg_background@2x.png")


if __name__ == "__main__":
    main()
