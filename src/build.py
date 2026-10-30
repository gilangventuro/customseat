"""Gabungkan template + foto + mask menjadi satu file HTML mandiri (bisa dibuka offline)."""
import base64, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
b64 = lambda p, mime: f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

html = (ROOT / "src/template.html").read_text()
html = (html.replace("__PHOTO__", b64(ROOT / "assets/foto-asli.webp", "image/webp"))
            .replace("__PA__", b64(ROOT / "src/masks/pack_a.png", "image/png"))
            .replace("__PB__", b64(ROOT / "src/masks/pack_b.png", "image/png"))
            .replace("__PC__", b64(ROOT / "src/masks/pack_c.png", "image/png"))
            .replace("__PL__", b64(ROOT / "src/masks/lum_base.jpg", "image/jpeg"))
            .replace("__PF__", b64(ROOT / "src/masks/lum_flat.jpg", "image/jpeg"))
            .replace("__PD__", b64(ROOT / "src/masks/pack_e.png", "image/png")))
out = ROOT / "simulator-warna-jok.html"
out.write_text(html)
print(out, f"{out.stat().st_size/1024:.0f} KB")
