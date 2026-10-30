"""Publish the 3D FBX viewer as dummy-kursi-gaming.html (replaces the old 2D multi-angle demo).

Unlike the other build_*.py scripts, this one does NOT inline the asset as base64 --
the FBX model (15MB) is fetched at runtime from assets/kursipijat.fbx via FBXLoader,
served as a plain static file by Vercel.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
src = ROOT / "src/template-3d.html"
out = ROOT / "dummy-kursi-gaming.html"
out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
print(out, f"{out.stat().st_size/1024:.1f} KB (+ assets/kursipijat.fbx served separately)")
