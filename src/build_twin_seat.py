"""
Proses assets/twin-seat.glb (hasil scan 3D, 1 mesh + 1 tekstur atlas acak)
menjadi assets/twin-seat-lowpoly.glb: mesh disederhanakan (decimate) dan
dipecah jadi beberapa sub-mesh berwarna polos berdasarkan warna di tekstur
aslinya, supaya tiap bagian bisa diberi material terpisah yang bisa
diganti warnanya di viewer three.js (seperti model kursi lain).

Jalankan: python src/build_twin_seat.py
"""
import numpy as np
import trimesh
from PIL import Image

SRC_GLB = "assets/twin-seat.glb"
OUT_GLB = "assets/twin-seat-lowpoly.glb"
TARGET_TOTAL_FACES = 140_000  # total segitiga gabungan setelah decimate


def rgb_to_hsv_np(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))
    rc = np.where(delta == 0, 0, (maxc - r) / np.where(delta == 0, 1, delta))
    gc = np.where(delta == 0, 0, (maxc - g) / np.where(delta == 0, 1, delta))
    bc = np.where(delta == 0, 0, (maxc - b) / np.where(delta == 0, 1, delta))
    h = np.where(maxc == r, (bc - gc), 0.0)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0, h)
    return h * 360, s, v


def classify_vec(hue, sat, val):
    # kalibrasi berdasar sampling tekstur nyata: titik gelap (AO baking)
    # harus difilter via val>0.3 supaya noise JPEG hitam tidak kebaca "berwarna"
    labels = np.full(hue.shape, "leather_main", dtype=object)
    green = (hue >= 140) & (hue <= 195) & (sat > 0.15) & (val > 0.3)
    red = ((hue >= 330) | (hue <= 15)) & (sat > 0.2) & (val > 0.3)
    yellow = (hue >= 35) & (hue <= 65) & (sat > 0.2) & (val > 0.3)
    labels[yellow] = "logo_yellow"
    labels[red] = "hardware_red"
    labels[green] = "accent_green"
    return labels


def main():
    print("Memuat scene...")
    scene = trimesh.load(SRC_GLB, process=False)
    name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[name]
    print("vertices", mesh.vertices.shape, "faces", mesh.faces.shape)

    uv = mesh.visual.uv
    img = mesh.visual.material.baseColorTexture
    if img is None:
        img = Image.open("assets/_tmp_basecolor.jpg")
    img = img.convert("RGB")
    arr = np.asarray(img)
    H, W = arr.shape[0], arr.shape[1]

    face_uv = uv[mesh.faces].mean(axis=1)  # (F,2) centroid UV per face
    px = np.clip((face_uv[:, 0] * W).astype(int), 0, W - 1)
    # glTF V=0 di atas gambar
    py = np.clip((face_uv[:, 1] * H).astype(int), 0, H - 1)
    face_rgb = arr[py, px, :].astype(np.float32) / 255.0  # (F,3)

    print("Klasifikasi warna per segitiga...")
    hue_deg, sat, val = rgb_to_hsv_np(face_rgb)
    labels = classify_vec(hue_deg, sat, val)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} segitiga ({c/len(labels)*100:.1f}%)")

    out_scene = trimesh.Scene()
    for cluster in unique:
        face_mask = labels == cluster
        sub_faces = mesh.faces[face_mask]
        # warna representatif: ambil persentil-75 brightness supaya tidak
        # ikut area gelap bayangan hasil baking foto
        cluster_rgb = face_rgb[face_mask]
        bright = cluster_rgb.sum(axis=1)
        thresh = np.percentile(bright, 85)
        bright_pixels = cluster_rgb[bright >= thresh]
        rep_color = bright_pixels.mean(axis=0) if len(bright_pixels) else cluster_rgb.mean(axis=0)
        # tekstur hasil scan membawa baking shadow/AO ringan; angkat value
        # sedikit saja supaya tidak terlalu suram tanpa mengubah hue
        rh, rs, rv = rgb_to_hsv_np(rep_color.reshape(1, 1, 3))
        rv = np.clip(rv * 1.15 + 0.03, 0, 0.9)
        rs = np.clip(rs * 1.05, 0, 1)
        import colorsys as _cs
        rep_color = np.array(_cs.hsv_to_rgb(float(rh[0, 0]) / 360.0, float(rs[0, 0]), float(rv[0, 0])))
        # glTF baseColorFactor ada di ruang linear, sedangkan rep_color di
        # atas masih ruang sRGB (dari piksel JPEG) -> konversi dulu, kalau
        # tidak modelnya akan tampak pudar/putih di viewer (linear dibaca
        # seolah-olah sudah sRGB lalu di-encode sRGB lagi saat ditampilkan)
        def srgb_to_linear(c):
            c = np.clip(c, 0, 1)
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        hex_preview = "#" + "".join(f"{int(np.clip(c,0,1)*255):02x}" for c in rep_color)
        rep_color = srgb_to_linear(rep_color)

        sub = trimesh.Trimesh(vertices=mesh.vertices, faces=sub_faces, process=False)
        sub.remove_unreferenced_vertices()

        face_ratio = TARGET_TOTAL_FACES * (face_mask.sum() / len(labels)) / max(len(sub.faces), 1)
        target_faces = max(200, int(len(sub.faces) * min(face_ratio, 1.0)))
        if target_faces < len(sub.faces):
            try:
                sub = sub.simplify_quadric_decimation(face_count=target_faces)
            except Exception as e:
                print(f"  (decimation {cluster} dilewati: {e})")

        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*rep_color.tolist(), 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.55,
            name=cluster,
        )
        sub.visual = trimesh.visual.ColorVisuals(mesh=sub, face_colors=None)
        sub.visual.material = mat
        print(f"  -> {cluster}: {len(sub.faces)} segitiga setelah decimate, warna {hex_preview}")
        out_scene.add_geometry(sub, node_name=cluster, geom_name=cluster)

    out_scene.export(OUT_GLB)
    import os
    print("Selesai ->", OUT_GLB, f"{os.path.getsize(OUT_GLB)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
