"""
Pecah assets/twin-seat-meshy.glb (1 mesh gabungan hasil Meshy AI, 2 kursi
captain bersebelahan) menjadi 3 sub-mesh berdasarkan jenis permukaan:
  - leather_main    : kulit polos (bolster samping, dudukan tepi, dll)
  - leather_pattern : panel berlian/quilted (bagian tengah sandaran, dudukan,
                       footrest) -- warnanya SAMA PERSIS dengan leather_main
                       di tekstur asli, cuma beda motif jahitan (emboss),
                       jadi tidak bisa dipisah lewat warna.
  - leather_accent  : list/piping hijau -- dideteksi lewat warna (hue hijau).

Ketiganya berlaku SIMETRIS di kursi kiri & kanan (bukan per-sisi).

Cara deteksi panel berlian: garis jahitan (kerapatan tekstur tinggi di UV)
dipakai sebagai "dinding" topologi di graf ketetanggaan segitiga (face
adjacency) -- BUKAN sebagai kelas warna final. Area kulit yang dikelilingi
jahitan rapat di semua sisi (= komponen terhubung kecil, karena terpotong-
potong jadi sel-sel berlian) diklasifikasi "pattern". Area kulit polos yang
menyambung luas tanpa banyak sekat jahitan (bolster, tepi dudukan) tetap jadi
satu komponen besar -> "main". Garis jahitan itu sendiri ikut ke kelas
tetangga mayoritasnya di langkah akhir.

Alur:
  1. src/convert_twin_meshy.py -> assets/twin-seat-meshy.glb (1 material).
  2. src/_export_merged_nodraco.py (Blender) -> _tmp_merged_nodraco.glb
     (tanpa draco supaya trimesh bisa baca UV+tekstur).
  3. Skrip ini (trimesh+numpy+PIL+networkx) -> _tmp_classified.glb (3
     material, masih tanpa draco).
  4. src/_compress_final.py (Blender) -> assets/twin-seat-meshy.glb final
     (re-import + export draco, supaya file kecil).

Jalankan:
  python src/split_twin_seats.py
"""
import numpy as np
import trimesh
import networkx as nx

SRC_GLB = "_tmp_merged_nodraco.glb"
OUT_GLB = "_tmp_classified.glb"

PATCH = 14           # ukuran patch (px) utk ukur kerapatan jahitan di UV
STITCH_STD_THRESH = 17.0   # ambang stddev lokal = dianggap "garis jahitan"
MAX_PATTERN_COMPONENT = 900  # komponen plain seluas ini atau lebih kecil -> panel berlian
MIN_ISLAND = 40       # island lebih kecil dari ini (hasil akhir) digabung ke tetangga mayoritas


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


def box_stats(ii, ii2, cy, cx, half):
    y0, y1 = cy, cy + 2 * half + 1
    x0, x1 = cx, cx + 2 * half + 1
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    s2 = ii2[y1, x1] - ii2[y0, x1] - ii2[y1, x0] + ii2[y0, x0]
    n = (y1 - y0) * (x1 - x0)
    mean = s / n
    var = np.maximum(s2 / n - mean ** 2, 0)
    return np.sqrt(var)


def main():
    print("Load mesh...")
    scene = trimesh.load(SRC_GLB, process=False)
    name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[name]
    uv = mesh.visual.uv
    img = mesh.visual.material.baseColorTexture.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    H, W = arr.shape[0], arr.shape[1]
    gray = arr.mean(axis=-1)

    print("Hitung UV & warna rata-rata tiap segitiga...")
    face_uv = uv[mesh.faces].mean(axis=1)
    px = np.clip((face_uv[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip(((1 - face_uv[:, 1]) * H).astype(int), 0, H - 1)
    face_rgb = arr[py, px, :] / 255.0
    hue, sat, val = rgb_to_hsv_np(face_rgb)

    print("Hitung kerapatan tekstur (local stddev) di sekitar tiap UV...")
    pad = PATCH // 2
    gray_pad = np.pad(gray, pad, mode="edge").astype(np.float64)
    ii = np.pad(np.cumsum(np.cumsum(gray_pad, axis=0), axis=1), ((1, 0), (1, 0)))
    ii2 = np.pad(np.cumsum(np.cumsum(gray_pad * gray_pad, axis=0), axis=1), ((1, 0), (1, 0)))
    local_std = box_stats(ii, ii2, py, px, pad)

    green = (hue >= 130) & (hue <= 200) & (sat > 0.12) & (val > 0.15)
    stitch = (~green) & (local_std > STITCH_STD_THRESH)
    plain = ~green & ~stitch

    n = len(mesh.faces)
    print(f"  green: {green.sum()} ({green.sum()/n*100:.1f}%)  stitch: {stitch.sum()} ({stitch.sum()/n*100:.1f}%)  plain: {plain.sum()} ({plain.sum()/n*100:.1f}%)")

    print("Cari komponen terhubung di antara segitiga 'plain' (dibatasi garis jahitan & aksen)...")
    adjacency = mesh.face_adjacency
    both_plain = plain[adjacency[:, 0]] & plain[adjacency[:, 1]]
    edges = adjacency[both_plain]
    g = nx.Graph()
    g.add_nodes_from(np.where(plain)[0])
    g.add_edges_from(edges)
    components = list(nx.connected_components(g))
    print(f"  {len(components)} komponen plain ditemukan")

    labels = np.full(n, "", dtype=object)
    labels[green] = "leather_accent"
    comp_sizes = []
    for comp in components:
        comp = list(comp)
        comp_sizes.append(len(comp))
        cls = "leather_pattern" if len(comp) <= MAX_PATTERN_COMPONENT else "leather_main"
        labels[comp] = cls
    comp_sizes = np.array(comp_sizes)
    print(f"  ukuran komponen: min={comp_sizes.min()} max={comp_sizes.max()} median={int(np.median(comp_sizes))}")

    print("Assign segitiga 'stitch' ke kelas tetangga mayoritas...")
    face_adj_list = [[] for _ in range(n)]
    for a, b in mesh.face_adjacency:
        face_adj_list[a].append(b)
        face_adj_list[b].append(a)
    stitch_idx = np.where(stitch)[0]
    # iterasi beberapa kali supaya stitch yg tetangganya jg stitch ikut terselesaikan
    for _ in range(4):
        changed = 0
        for i in stitch_idx:
            if labels[i]:
                continue
            votes = {}
            for j in face_adj_list[i]:
                lbl = labels[j]
                if lbl:
                    votes[lbl] = votes.get(lbl, 0) + 1
            if votes:
                labels[i] = max(votes, key=votes.get)
                changed += 1
        if changed == 0:
            break
    labels[labels == ""] = "leather_main"

    for lbl in np.unique(labels):
        c = (labels == lbl).sum()
        print(f"  {lbl}: {c} segitiga ({c/n*100:.1f}%)")

    print("Bersihkan noise (island kecil) via connected components (final)...")
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    edges2 = adjacency[same]
    g2 = nx.Graph()
    g2.add_nodes_from(range(n))
    g2.add_edges_from(edges2)
    final_components = list(nx.connected_components(g2))
    clean = labels.copy()
    for comp in final_components:
        comp = list(comp)
        lbl = labels[comp[0]]
        if len(comp) < MIN_ISLAND:
            # gabung ke tetangga mayoritas (di luar komponen ini)
            votes = {}
            for i in comp:
                for j in face_adj_list[i]:
                    if j not in comp:
                        votes[labels[j]] = votes.get(labels[j], 0) + 1
            if votes:
                clean[comp] = max(votes, key=votes.get)
    for lbl in np.unique(clean):
        c = (clean == lbl).sum()
        print(f"  (bersih) {lbl}: {c} segitiga ({c/n*100:.1f}%)")

    print("Export GLB 3-material...")
    out_scene = trimesh.Scene()
    src_mat = mesh.visual.material
    for cluster in np.unique(clean):
        face_mask = clean == cluster
        sub_faces = mesh.faces[face_mask]
        sub = trimesh.Trimesh(vertices=mesh.vertices, faces=sub_faces, process=False)
        sub.visual = trimesh.visual.TextureVisuals(uv=uv, material=src_mat.copy())
        sub.visual.material.name = cluster
        sub.remove_unreferenced_vertices()
        out_scene.add_geometry(sub, node_name=cluster, geom_name=cluster)

    out_scene.export(OUT_GLB, include_normals=True)
    import os
    print("Selesai ->", OUT_GLB, f"{os.path.getsize(OUT_GLB)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
