"""Embed multi-angle photos + masks into one standalone HTML file (dummy multi-angle demo)."""
import base64, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENES_OUT = ROOT / "src/scenes/out"
b64 = lambda p, mime: f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

html = (ROOT / "src/template-multiangle.html").read_text(encoding="utf-8")
for name in ["depan", "samping", "belakang", "atas"]:
    html = (html.replace(f"__{name.upper()}_PHOTO__",   b64(SCENES_OUT / f"{name}_photo.jpg", "image/jpeg"))
                .replace(f"__{name.upper()}_MASK__",    b64(SCENES_OUT / f"{name}_mask.png", "image/png"))
                .replace(f"__{name.upper()}_ARMREST__", b64(SCENES_OUT / f"{name}_armrest.png", "image/png"))
                .replace(f"__{name.upper()}_LUM__",     b64(SCENES_OUT / f"{name}_lum.jpg", "image/jpeg")))

out = ROOT / "dummy-kursi-gaming.html"
out.write_text(html, encoding="utf-8")
print(out, f"{out.stat().st_size/1024:.0f} KB")
