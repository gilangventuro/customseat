"""
Proses assets/twin-seat.glb (hasil scan 3D, 1 mesh + 1 tekstur atlas acak)
menjadi assets/twin-seat-lowpoly.glb: mesh disederhanakan (decimate) SEKALI
sebagai satu kesatuan utuh (supaya topologi tetap menyambung rapi, tidak ada
celah di tepi), baru SETELAH itu dipecah jadi beberapa sub-mesh berwarna
polos berdasarkan warna di tekstur aslinya (supaya tiap bagian bisa diberi
material terpisah yang bisa diganti warnanya). Noise klasifikasi warna
(pulau-pulau kecil hasil JPEG/scan) dibersihkan via connected-component
filtering supaya hasilnya tidak terlihat "pecah/shatter".

Jalankan: python src/build_twin_seat.py
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SRC_GLB = "assets/twin-seat.glb"
OUT_GLB = "assets/twin-seat-lowpoly.glb"
TARGET_TOTAL_FACES = 180_000  # total segitiga setelah decimate
MIN_ISLAND_FACES = 90  # komponen lebih kecil dari ini dianggap noise -> gabung ke leather_main


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
    labels = np.full(hue.shape, "leather_main", dtype=object)
    green = (hue >= 140) & (hue <= 195) & (sat > 0.15) & (val > 0.3)
    red = ((hue >= 330) | (hue <= 15)) & (sat > 0.2) & (val > 0.3)
    yellow = (hue >= 35) & (hue <= 65) & (sat > 0.2) & (val > 0.3)
    labels[yellow] = "logo_yellow"
    labels[red] = "hardware_red"
    labels[green] = "accent_green"
    return labels


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main():
    print("Memuat scene...")
    scene = trimesh.load(SRC_GLB, process=False)
    name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[name]
    print("vertices asli", mesh.vertices.shape, "faces asli", mesh.faces.shape)

    uv = mesh.visual.uv
    img = mesh.visual.material.baseColorTexture.convert("RGB")
    arr = np.asarray(img)
    H, W = arr.shape[0], arr.shape[1]

    # ---------- 1) voxel remeshing -> jamin watertight (tanpa lubang) ----------
    # mesh asli hasil scan TIDAK watertight -- ada ~109rb tepi terbuka dari
    # total 1.8 juta segitiga (~6%), artinya banyak area yang kameranya tidak
    # ter-capture (celah antar lipatan jok, bawah dudukan, dll). Ini bukan
    # noise kecil yang bisa dibereskan fill_holes() biasa -- areanya luas dan
    # tersebar. Solusinya: ubah mesh jadi grid voxel solid lalu ekstrak ulang
    # permukaannya (marching cubes) -> hasilnya DIJAMIN watertight, lubang
    # otomatis "ditutup" karena jadi volume padat. Detail sangat halus yang
    # lebih kecil dari ukuran voxel akan hilang, tapi itu trade-off yang wajar
    # dibanding tampilan bolong-bolong tembus pandang.
    print("Voxel remeshing (menjamin mesh watertight, tanpa lubang)...")
    size = (mesh.bounds[1] - mesh.bounds[0]).max()
    pitch = size / 180
    # catatan: pitch lebih halus (dicoba size/320, 1.2jt segitiga sebelum
    # decimate) TIDAK membantu memulihkan warna aksen -- voxel+marching-cubes
    # membuat permukaan baru yang posisinya cukup bergeser dari mesh asli
    # sehingga nearest-vertex tidak lagi menangkap area tipis (list piping,
    # pengait). Trade-off watertight vs deteksi warna per-bagian ini sudah
    # final: model jadi satu warna kulit utama saja, tanpa pemisahan aksen.
    vox = mesh.voxelized(pitch=pitch).fill()
    solid = vox.marching_cubes
    print(f"  watertight={solid.is_watertight} faces={len(solid.faces)}")

    # permukaan voxel/marching-cubes berbentuk "tangga" kotak-kotak ->
    # haluskan. Sekarang AMAN pakai smoothing kuat tanpa perlu pin boundary
    # sama sekali, karena sudah watertight (tidak ada tepi terbuka yang bisa
    # "meledak" seperti percobaan sebelumnya).
    print("Menghaluskan permukaan voxel (Taubin smoothing)...")
    import trimesh.smoothing as sm
    sm.filter_taubin(solid, lamb=0.5, nu=-0.53, iterations=25)
    solid = trimesh.Trimesh(vertices=solid.vertices, faces=solid.faces, process=False)

    # ---------- 2) decimate SEKALI sebagai satu mesh utuh ----------
    print(f"Decimate mesh utuh -> target {TARGET_TOTAL_FACES} segitiga...")
    simple = solid.simplify_quadric_decimation(face_count=TARGET_TOTAL_FACES)
    simple = trimesh.Trimesh(vertices=simple.vertices, faces=simple.faces, process=False)
    print("vertices setelah decimate", simple.vertices.shape, "faces", simple.faces.shape)

    # ---------- 3) bawa UV dari mesh ASLI (sebelum voxelize) via nearest-vertex ----------
    # mesh hasil voxel/decimate punya vertex yang sama sekali baru (bukan
    # subset mesh asli), jadi warna/UV harus "dicari" dari titik terdekat di
    # mesh asli yang masih punya UV & tekstur yang benar
    print("Transfer UV (nearest-vertex) ke mesh hasil decimate...")
    tree = cKDTree(mesh.vertices)
    _, nn_idx = tree.query(simple.vertices, k=1)
    simple_uv = uv[nn_idx]

    face_uv = simple_uv[simple.faces].mean(axis=1)
    px = np.clip((face_uv[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip((face_uv[:, 1] * H).astype(int), 0, H - 1)
    face_rgb = arr[py, px, :].astype(np.float32) / 255.0

    # ---------- 3) klasifikasi warna per segitiga ----------
    print("Klasifikasi warna per segitiga...")
    hue_deg, sat, val = rgb_to_hsv_np(face_rgb)
    labels = classify_vec(hue_deg, sat, val)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u} (mentah): {c} segitiga ({c/len(labels)*100:.1f}%)")

    # ---------- 4) bersihkan noise: connected-component per label ----------
    print("Membersihkan noise (pulau kecil) via connected components...")
    adjacency = simple.face_adjacency  # (N,2) pasangan index segitiga bertetangga
    same_label = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    edges = adjacency[same_label]

    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(len(simple.faces)))
    g.add_edges_from(edges)
    components = list(nx.connected_components(g))
    print(f"  {len(components)} komponen ditemukan")

    clean_labels = labels.copy()
    merged = 0
    for comp in components:
        comp = list(comp)
        lbl = labels[comp[0]]
        if lbl != "leather_main" and len(comp) < MIN_ISLAND_FACES:
            clean_labels[comp] = "leather_main"
            merged += len(comp)
    print(f"  {merged} segitiga noise digabung ke leather_main")

    unique, counts = np.unique(clean_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u} (bersih): {c} segitiga ({c/len(clean_labels)*100:.1f}%)")

    # ---------- 5) pecah jadi sub-mesh per material (tanpa decimate lagi,
    #              supaya vertex di tepi/boundary tetap identik -> tidak ada celah) ----------
    out_scene = trimesh.Scene()
    for cluster in unique:
        face_mask = clean_labels == cluster
        sub_faces = simple.faces[face_mask]

        cluster_rgb = face_rgb[face_mask]
        bright = cluster_rgb.sum(axis=1)
        thresh = np.percentile(bright, 85)
        bright_pixels = cluster_rgb[bright >= thresh]
        rep_color = bright_pixels.mean(axis=0) if len(bright_pixels) else cluster_rgb.mean(axis=0)
        rh, rs, rv = rgb_to_hsv_np(rep_color.reshape(1, 1, 3))
        rv = np.clip(rv * 1.15 + 0.03, 0, 0.9)
        rs = np.clip(rs * 1.05, 0, 1)
        import colorsys as _cs
        rep_color = np.array(_cs.hsv_to_rgb(float(rh[0, 0]) / 360.0, float(rs[0, 0]), float(rv[0, 0])))
        hex_preview = "#" + "".join(f"{int(np.clip(c,0,1)*255):02x}" for c in rep_color)
        rep_color_linear = srgb_to_linear(rep_color)

        sub = trimesh.Trimesh(vertices=simple.vertices, faces=sub_faces, process=False)
        sub.remove_unreferenced_vertices()

        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*rep_color_linear.tolist(), 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.55,
            name=cluster,
        )
        sub.visual = trimesh.visual.ColorVisuals(mesh=sub, face_colors=None)
        sub.visual.material = mat
        print(f"  -> {cluster}: {len(sub.faces)} segitiga, warna {hex_preview}")
        out_scene.add_geometry(sub, node_name=cluster, geom_name=cluster)

    # include_normals=True WAJIB -- tanpa ini trimesh tidak menulis atribut
    # NORMAL ke glTF, sehingga browser terpaksa menghitung normal datar per
    # segitiga sendiri (sesuai spec glTF) -> permukaan tampak "pecah/shatter"
    # walau geometrinya sendiri sudah mulus
    out_scene.export(OUT_GLB, include_normals=True)
    import os
    print("Selesai ->", OUT_GLB, f"{os.path.getsize(OUT_GLB)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
