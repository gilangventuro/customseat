"""Publish the model-selection carousel as pilih-model.html.

Same pattern as build_3d.py: no base64 inlining, FBX is fetched at runtime
from assets/kursipijat.fbx (shared/cloned across the carousel cards).
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
src = ROOT / "src/template-select.html"
out = ROOT / "pilih-model.html"
out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
print(out, f"{out.stat().st_size/1024:.1f} KB (+ assets/kursipijat.fbx served separately)")
