#!/usr/bin/env python3
"""Convert an X Window Dump (.xwd, produced by `xwd`) to PNG using only PIL.

Handles the common Xvfb case: TrueColor ZPixmap, depth 24 / 32 bpp, with RGB
masks in the header. No ImageMagick / netpbm needed.

Usage:  python3 xwd2png.py screen.xwd screen.png
"""
import struct
import sys
from PIL import Image

HDR_FIELDS = 25  # CARD32 header fields (sz_XWDheader = 100 bytes)


def low_bit(mask):
    if mask == 0:
        return 0
    s = 0
    while not (mask >> s) & 1:
        s += 1
    return s


def mask_width(mask, shift):
    w = 0
    m = mask >> shift
    while (m >> w) & 1:
        w += 1
    return w


def main():
    src, out = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()

    # Detect endianness from file_version (field 1 must be 7).
    for endian in (">", "<"):
        hdr = struct.unpack(endian + "I" * HDR_FIELDS, data[:4 * HDR_FIELDS])
        if hdr[1] == 7:
            break
    else:
        raise SystemExit("not an XWD file (bad version)")

    header_size = hdr[0]
    pix_format = hdr[2]
    width = hdr[4]
    height = hdr[5]
    bpp = hdr[11]
    bytes_per_line = hdr[12]
    red_mask, green_mask, blue_mask = hdr[14], hdr[15], hdr[16]
    ncolors = hdr[19]

    pixels_off = header_size + ncolors * 12  # skip window name + colormap
    pix = data[pixels_off:]

    if pix_format != 2 or bpp not in (24, 32):
        raise SystemExit(f"unsupported xwd: format={pix_format} bpp={bpp}")

    rs, gs, bs = low_bit(red_mask), low_bit(green_mask), low_bit(blue_mask)
    step = bpp // 8
    img = Image.new("RGB", (width, height))
    px = img.load()
    # pixel CARD32 order in the file follows the header byte order we detected
    p32 = ">I" if endian == ">" else "<I"

    for y in range(height):
        row = pix[y * bytes_per_line: y * bytes_per_line + width * step]
        for x in range(width):
            chunk = row[x * step:x * step + step]
            if step == 3:
                chunk = chunk + b"\x00"
            (val,) = struct.unpack(p32, chunk)
            r = (val & red_mask) >> rs
            g = (val & green_mask) >> gs
            b = (val & blue_mask) >> bs
            px[x, y] = (r & 0xFF, g & 0xFF, b & 0xFF)

    img.save(out)
    print(f"wrote {out}  ({width}x{height}, bpp={bpp})")


if __name__ == "__main__":
    main()
