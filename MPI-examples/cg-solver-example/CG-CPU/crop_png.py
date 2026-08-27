#!/usr/bin/env python3
"""Crop a PNG to a box. Usage: crop_png.py in.png out.png L T R B"""
import sys
from PIL import Image
img = Image.open(sys.argv[1])
L, T, R, B = (int(v) for v in sys.argv[3:7])
img.crop((L, T, R, B)).save(sys.argv[2])
print(f"cropped {sys.argv[1]} -> {sys.argv[2]} ({R-L}x{B-T})")
