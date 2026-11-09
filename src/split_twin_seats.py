"""
Pecah assets/twin-seat-meshy.glb (1 mesh gabungan hasil Meshy AI, 2 kursi
captain bersebelahan) menjadi 3 sub-mesh berdasarkan jenis permukaan:
  - leather_main    : kulit polos (bolster samping, headrest atas, dll)
  - leather_pattern : panel berlian/quilted (bagian tengah sandaran, dudukan,
                       footrest) -- warnanya SAMA PERSIS dengan leather_main
                       di tekstur asli, cuma beda motif jahitan (emboss),
                       jadi tidak bisa dipisah lewat warna.
  - leather_accent  : list/piping -- dideteksi lewat warna (hue hijau/teal).

Ketiganya berlaku SIMETRIS di kursi kiri & kanan (bukan per-sisi).

RIWAYAT PENDEKATAN (penting dibaca sebelum ubah threshold di bawah):
Percobaan pertama menganalisis kerapatan tekstur langsung di ruang UV (pakai
box-stddev di sekitar koordinat UV tiap segitiga). Hasilnya kacau, sebab
tekstur model ini pakai UV atlas HASIL SCAN -- potongan-potongan kecil
disusun acak tanpa padding (lihat assets kalau mau cek sendiri), jadi "area
sekitar" satu titik UV sering kali sebenarnya milik permukaan 3D lain yang
tidak berhubungan. Percobaan kedua (ukuran island via face_adjacency) juga
gagal karena mesh-nya sendiri sudah terpecah ~2000 potongan kecil terputus
(bukan 1 permukaan menyambung) akibat proses rekonstruksi/decimation, jadi
ukuran potongan tidak berkorelasi dengan "polos vs berlian". Percobaan
ketiga (kelengkungan normal 3D) juga gagal karena hampir semua permukaan,
termasuk yang polos, punya noise kecil dari hasil scan.

Yang akhirnya berhasil: render tampilan depan model (kamera tetap), analisis
kerapatan tepi/jahitan di gambar HASIL RENDER itu (bukan tekstur UV mentah --
di ruang gambar hasil render, piksel yang berdekatan memang berdekatan di
model asli, beda dari atlas UV yang acak), lalu proyeksikan balik mask-nya
ke tiap segitiga lewat ray-casting dari kamera yang sama. Bagian yang tidak
kelihatan dari depan (sisi belakang headrest, dll) default ke leather_main
karena motif berlian di model ini memang cuma ada di permukaan yang
dihadapkan ke pengguna.

"Kelihatan dari depan" TIDAK pakai ray-casting ketat (sempat dicoba 2 cara:
trimesh+rtree murni Python terlalu lambat -- 150rb ray >40 menit belum
selesai; BVHTree Blender lebih cepat tapi "first-hit" terlalu ketat --
segitiga di sisi lekukan motif berlian ikut dibuang karena ketutup puncak
jahitan di sebelahnya, padahal jelas bagian dari zona motif). Sebagai
gantinya dipakai toleransi kedalaman per-piksel (lihat DEPTH_TOLERANCE):
segitiga dianggap "dekat permukaan depan" kalau depth-nya tidak jauh dari
permukaan TERDEPAN di piksel proyeksi yang sama -- jauh lebih murah (tidak
perlu BVH/ray-cast sama sekali, cuma scatter-min numpy) dan hasilnya lebih
sesuai maksud ("se-zona", bukan "triangle paling depan secara harfiah").

Alur:
  1. src/convert_twin_meshy.py (Blender) -> assets/twin-seat-meshy.glb (1 material).
  2. src/_export_merged_nodraco.py (Blender) -> _tmp_merged_nodraco.glb
     (tanpa draco supaya trimesh bisa baca UV+tekstur).
  3. src/_render_front.py (Blender) -> _tmp_front.png + _tmp_cam.json.
  4. Skrip ini (trimesh+numpy+scipy) -> _tmp_classified.glb (3
     material, masih tanpa draco).
  5. src/_compress_final.py (Blender) -> assets/twin-seat-meshy.glb final
     (re-import + export draco, supaya file kecil).

Jalankan:
  python src/split_twin_seats.py
"""
import json
import os

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

SRC_GLB = "_tmp_merged_nodraco.glb"
FRONT_PNG = "_tmp_front.png"
CAM_JSON = "_tmp_cam.json"
OUT_GLB = "_tmp_classified.glb"
TINT_BASE_PNG = "assets/textures/leather-tint-base.webp"  # lihat brighten_for_tint

# -- parameter "terangkan tekstur" biar tint warna customizer tidak gelap --
BRIGHTEN_LO_PCT = 5      # percentile bawah tekstur asli dipakai sbg acuan "gelap"
BRIGHTEN_HI_PCT = 90     # percentile atas tekstur asli dipakai sbg acuan "terang"
                         # (bukan 99/100 -- highlight spekular ekstrem di beberapa
                         # piksel jangan sampai menekan rentang stretch)
BRIGHTEN_TARGET_LO = 215  # hasil stretch: piksel paling gelap jadi seterang ini
BRIGHTEN_TARGET_HI = 255  # & piksel paling terang jadi seterang ini (0-255)

SPATIAL_VOTE_K = 24            # jumlah tetangga spasial (KD-tree) terdekat buat voting
                               # (sebelum difilter same_side, lihat bawah -- perlu agak
                               # banyak krn sebagian bakal kebuang gara2 filter sisi shell)
SPATIAL_VOTE_DOMINANCE = 0.6   # cuma balik label kalau mayoritas tetangga >= ini (lihat bawah)
SPATIAL_VOTE_PASSES = 6        # diulang beberapa kali, longgar per-putaran -- lihat komentar
                               # di pemanggilannya kenapa ini lebih baik drpd 1x ambang tinggi

# -- parameter deteksi area motif berlian dari render depan --
EDGE_STD_PATCH = 10       # radius box utk ukur stddev lokal (deteksi garis jahitan)
EDGE_STD_THRESH = 8.0     # stddev lokal di atas ini = garis jahitan
DENSITY_RADIUS = 20       # radius box utk ukur "berapa rapat garis jahitan di sekitar sini"
DENSITY_THRESH = 0.35     # kerapatan di atas ini = garis outline motif berlian
GAP_CLOSE = 13            # radius dilasi utk nyambungin celah kecil antar segmen garis
                          # sebelum fill-holes (lihat build_pattern_mask_from_render)
MASK_SMOOTH_SIGMA = 10    # sigma Gaussian blur buat membulatkan sudut blocky hasil
                          # dilasi/erosi kotak di atas (lihat build_pattern_mask_from_render)
STRIP_TENDRIL_RADIUS = 23 # radius opening (erosi lalu dilasi) utk motong "juntaian"/tendril
                          # tipis yang masih NYAMBUNG ke blob utama (makanya lolos dari filter
                          # ukuran blob -- dihitung 1 komponen besar yang sama). Tendril ini
                          # yang bikin tepi pattern/main kelihatan "robek kertas" di 3D.
MAX_EDGE_SUBDIV = 0.0012  # satuan dunia (lihat main()) -- pecah segitiga lebih besar dari ini
                          # sebelum klasifikasi, biar tepi pattern/main halus bukan blocky
MIN_PATTERN_BLOB = 20000  # blob hasil klasifikasi lebih kecil dari ini dibuang (bukan panel asli)
MAX_HOLE_FILL = 45000     # lubang di dalam blob pattern lebih kecil dari ini diisi -- dinaikkan
                          # jauh drpd semula (2500) krn alur jahitan/lekukan RECESSED di
                          # DALAM panel berlian (antar tuft) ikut ke-deteksi sbg "garis
                          # terpisah" yg bikin ratusan lubang kecil-menengah (sampai ~39rb px)
                          # tersebar di SELURUH interior panel -- inilah sumber noise "robek
                          # kertas" pas tint diaplikasikan (segitiga di sekitar lekukan jahitan
                          # itu gampang kepilih acak main/pattern). Lekukan ini bagian dari
                          # SATU motif berlian yg sama, bukan area "main" terpisah, jadi wajar
                          # diisi penuh. Komponen "main" asli (headrest, bolster, dll) jauh
                          # lebih besar dari ini (~2 juta px) jadi aman tidak ikut keisi.


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


def box_filter_sum(binary_img, radius):
    """Jumlah piksel 'true' dalam box (2r+1)x(2r+1) di sekitar tiap piksel."""
    H, W = binary_img.shape
    p = np.pad(binary_img.astype(np.float64), radius, mode="edge")
    ii = np.pad(np.cumsum(np.cumsum(p, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y0, y1 = ys, ys + 2 * radius + 1
    x0, x1 = xs, xs + 2 * radius + 1
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def build_pattern_mask_from_render():
    """Hasilkan mask 2D (di ruang gambar render depan) area panel berlian.

    Pendekatan AWAL (kerapatan garis jahitan lokal lalu threshold langsung)
    ternyata cuma melacak GARIS jahitannya sendiri (thin outline mengikuti
    pola berlian), bukan AREA penuh di dalamnya -- hasil threshold density
    juga ikut menyala di sepanjang garis jahitan tunggal bolster/headrest
    (yang notabene cuma 1-2 garis panjang, bukan motif berlian), jadi area
    yang kena jauh lebih luas dari panel berlian asli (nyaris seluruh jok).

    Pendekatan yang benar: deteksi garis jahitan (seperti di atas) itu SUDAH
    menghasilkan outline tertutup (closed loop) di sekeliling tiap sel
    berlian -- makanya daripada threshold density dipakai langsung sbg mask
    area, outline itu di-dilasi dikit buat nyambungin celah kecil antar
    segmen garis, lalu di-ISI (binary_fill_holes) sehingga interior tiap sel
    berlian ikut terisi, baru di-erosi balik ke ukuran semula. Area bolster
    polos & headrest atas yang cuma punya 1-2 garis jahitan PANJANG (tidak
    membentuk loop tertutup) otomatis tidak ikut terisi karena bukan area
    "dikelilingi garis dari segala arah".
    """
    img = Image.open(FRONT_PNG).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    gray = arr.mean(axis=-1)
    H, W = gray.shape

    pad = EDGE_STD_PATCH // 2
    gp = np.pad(gray, pad, mode="edge").astype(np.float64)
    ii = np.pad(np.cumsum(np.cumsum(gp, axis=0), axis=1), ((1, 0), (1, 0)))
    ii2 = np.pad(np.cumsum(np.cumsum(gp * gp, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y0, y1 = ys, ys + 2 * pad + 1
    x0, x1 = xs, xs + 2 * pad + 1
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    s2 = ii2[y1, x1] - ii2[y0, x1] - ii2[y1, x0] + ii2[y0, x0]
    n = (y1 - y0) * (x1 - x0)
    mean = s / n
    local_std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    edge = local_std > EDGE_STD_THRESH

    bg_mask = (
        (np.abs(arr[:, :, 0] - 20) < 6)
        & (np.abs(arr[:, :, 1] - 23) < 6)
        & (np.abs(arr[:, :, 2] - 25) < 6)
        & (local_std < 2)
    )
    # bg_mask juga nyala di ribuan kantong bayangan KECIL di lekukan motif
    # berlian (celah antar tuft yang gelap & nyaris rata warnanya -- ikut
    # memenuhi kriteria "gelap & flat" di atas) -- bukan cuma di area
    # background asli di sekeliling siluet jok. Kantong kecil ini bikin
    # obj_mask bolong-bolong di tengah jok (noise "garam-merica" yang
    # akhirnya ikut keproyeksi ke 3D). Background ASLI itu satu komponen
    # raksasa yang ngelilingin siluet jok -- jadi cukup ambil komponen
    # terhubung TERBESAR sbg background, sisanya (kantong bayangan kecil)
    # dianggap bagian dari objek.
    bg_lbl, bg_nlab = ndimage.label(bg_mask, structure=np.ones((3, 3)))
    if bg_nlab > 0:
        bg_sizes = ndimage.sum(np.ones_like(bg_lbl), bg_lbl, index=np.arange(1, bg_nlab + 1))
        bg_main = np.argmax(bg_sizes) + 1
        bg_mask = bg_lbl == bg_main
    obj_mask = ~bg_mask

    density = box_filter_sum(edge, DENSITY_RADIUS) / (2 * DENSITY_RADIUS + 1) ** 2
    lines = (density > DENSITY_THRESH) & obj_mask
    lines = ndimage.binary_closing(lines, structure=np.ones((11, 11)))
    lines = ndimage.binary_opening(lines, structure=np.ones((7, 7)))

    closed = ndimage.binary_dilation(lines, structure=np.ones((GAP_CLOSE, GAP_CLOSE)))
    filled = ndimage.binary_fill_holes(closed)
    filled = ndimage.binary_erosion(filled, structure=np.ones((GAP_CLOSE, GAP_CLOSE)))

    # Potong "juntaian"/tendril tipis yang masih nempel di blob utama (garis
    # jahitan tunggal di bolster/headrest yang KEBETULAN nyambung ke tepi
    # panel berlian) -- opening (erosi lalu dilasi balik) dgn radius lebih
    # besar drpd lebar tendril akan memutusnya tanpa mengubah bentuk blob
    # utama yang solid. Ini beda dari filter ukuran blob di bawah, karena
    # tendril masih jadi SATU komponen terhubung dgn blob utama (makanya
    # lolos filter ukuran) -- harus diputus dulu secara geometris.
    filled = ndimage.binary_opening(
        filled, structure=np.ones((STRIP_TENDRIL_RADIUS, STRIP_TENDRIL_RADIUS))
    )

    # Dilasi/erosi pakai kernel KOTAK (np.ones) di atas ninggalin batas yang
    # blocky/zigzag (siku-siku & gerigi, bukan lengkung halus) -- begitu
    # diproyeksikan ke 3D jadi kelihatan "robek"/kurang rapi di sepanjang
    # tepi pattern vs main (lihat screenshot user). Haluskan dengan Gaussian
    # blur lalu threshold ulang di 0.5 -- cara umum buat membulatkan sudut
    # mask biner tanpa mengubah luas areanya secara signifikan.
    filled_f = ndimage.gaussian_filter(filled.astype(np.float32), sigma=MASK_SMOOTH_SIGMA)
    filled = filled_f > 0.5
    pat = filled & obj_mask

    lbl, nlab = ndimage.label(pat, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, nlab + 1))
    keep = np.zeros(nlab + 1, dtype=bool)
    keep[1:] = sizes > MIN_PATTERN_BLOB
    clean = keep[lbl]

    lbl2, nlab2 = ndimage.label(obj_mask & ~clean, structure=np.ones((3, 3)))
    sizes2 = ndimage.sum(np.ones_like(lbl2), lbl2, index=np.arange(1, nlab2 + 1))
    fillable = np.zeros(nlab2 + 1, dtype=bool)
    fillable[1:] = sizes2 < MAX_HOLE_FILL
    clean = clean | fillable[lbl2]

    clean = ndimage.binary_closing(clean, structure=np.ones((7, 7)))
    clean = ndimage.binary_opening(clean, structure=np.ones((5, 5)))
    return clean, obj_mask


DEPTH_TOLERANCE = 0.02  # satuan dunia -- toleransi "dekat permukaan terdepan" per
                         # piksel; harus > relief motif berlian tapi < ketebalan shell
FACING_THRESH = 0.3      # facing MENTAH harus < ini -- diketatkan lagi dari percobaan
                         # sebelumnya (0.65, dicoba session lalu, ternyata TERLALU
                         # longgar) -- sempat bikin ~38% dari segitiga yg lolos tes
                         # depth justru menghadap agak menjauh dari kamera (facing>0),
                         # termasuk bagian BAWAH/BELAKANG dudukan jok yang (krn shell
                         # tipis & DEPTH_TOLERANCE lebar) kebetulan depth-nya nyaris
                         # sama dgn permukaan depan di piksel yg sama -- hasilnya warna
                         # bocor ke rangka/bawah jok pas lihat dari belakang.
                         # tambahan di atas depth test (lihat komentar di atas)


def project_with_depth_tolerance(mesh, cam):
    """Proyeksikan centroid tiap segitiga ke piksel kamera depan. "Kelihatan"
    didefinisikan longgar: depth-nya dekat dengan permukaan paling depan di
    piksel yang sama (bukan harus PERSIS permukaan terdepan) -- supaya motif
    berlian yang py bergelombang 3D tidak kepotong oleh self-occlusion."""
    # matrix_world kamera diambil LANGSUNG dari Blender (lihat src/_render_front.py)
    # supaya tidak perlu reimplementasi konversi Euler->matrix yang rawan salah
    # urutan/konvensi -- kolom 0/1/2 dari bagian rotasi = sumbu local X/Y/Z kamera
    # di world space; kamera Blender melihat ke arah -Z lokal.
    #
    # PENTING: Blender otomatis konversi glTF (Y-up) -> Blender (Z-up) saat
    # import (bpy.ops.import_scene.gltf), tapi trimesh TIDAK -- trimesh.load()
    # di skrip ini membaca vertex mentah dalam konvensi Y-up asli glTF. Jadi
    # kamera (dari Blender, Z-up) harus dikonversi BALIK ke Y-up supaya cocok
    # dengan mesh.vertices di sini. Bug ini sempat bikin proyeksi acak-acakan
    # (segitiga di headrest/backrest proyeksi keluar frustum sama sekali).
    def blender_to_gltf(v):
        return np.array([v[0], v[2], -v[1]])

    M = np.array(cam["matrix_world"])
    loc = blender_to_gltf(M[:3, 3])
    right = blender_to_gltf(M[:3, 0])
    up = blender_to_gltf(M[:3, 1])
    forward = blender_to_gltf(-M[:3, 2])

    W, H = cam["res"]
    fov = cam["angle"]
    tan_half = np.tan(fov / 2)

    centroids = mesh.triangles_center
    normals = mesh.face_normals
    rel = centroids - loc
    depth = rel @ forward
    lateral = rel @ right
    vertical = rel @ up
    ndc_x = (lateral / depth) / tan_half
    ndc_y = (vertical / depth) / tan_half
    px = ((ndc_x * 0.5 + 0.5) * W).astype(int)
    py = ((1 - (ndc_y * 0.5 + 0.5)) * H).astype(int)
    # "Kelihatan dari depan" butuh DUA tes, bukan satu:
    #   1. DEPTH per-piksel (scatter-MIN) -- shell jok ini tipis, jadi sisi
    #      BELAKANG (backrest shell, beda permukaan sama sekali) proyeksinya
    #      sering jatuh di piksel yang SAMA dengan permukaan depan (dilihat
    #      lurus dari kamera depan, depan & belakang shell nyaris segaris).
    #      Tanpa depth test, segitiga belakang numpang ambil label mask yang
    #      sama sekali tidak relevan buat mereka (lihat render debug "dari
    #      belakang" -- penuh noise acak kalau depth test ditiadakan).
    #      Toleransinya (DEPTH_TOLERANCE) dibuat GENEROUS -- lebih lebar dari
    #      relief motif berlian (tonjolan/lekukan jahitan individual, biar
    #      semua "lapisan depan" tetap lolos) tapi masih lebih sempit dari
    #      ketebalan shell (biar lapisan belakang yg jelas beda permukaan
    #      tetap kebuang).
    #   2. FACING mentah per-segitiga (longgar) -- cuma buat jaga-jaga
    #      exclude segitiga yg depth-nya kebetulan lolos tapi normalnya
    #      sudah jelas menghadap berlawanan arah kamera.
    # (Threshold facing MENTAH/ketat sendirian pernah dicoba dan gagal --
    # relief motif berlian bikin normal lompat2 antar tetangga; tapi itu
    # sebelum ketahuan akar masalah utamanya ada di langkah cleanup/voting,
    # bukan di sini -- lihat catatan di bagian voting mayoritas spasial.)
    facing = normals @ forward  # < 0 berarti normal menghadap ke arah kamera
    in_frustum = (depth > 1e-6) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    dmin = np.full((H, W), np.inf)
    valid_idx = np.where(in_frustum)[0]
    np.minimum.at(dmin, (py[valid_idx], px[valid_idx]), depth[valid_idx])

    near_front = np.zeros(len(centroids), dtype=bool)
    near_front[valid_idx] = (
        (depth[valid_idx] <= dmin[py[valid_idx], px[valid_idx]] + DEPTH_TOLERANCE)
        & (facing[valid_idx] < FACING_THRESH)
    )

    return px, py, near_front


def brighten_for_tint(arr):
    """Terangkan tekstur asli (kulit hitam gelap, mean cuma ~45/255) supaya
    tint warna di customizer (material.color dikalikan ke tekstur, lihat
    dummy-kursi-twinseat.html) tidak selalu jatuh gelap.

    three.js menerapkan warna pilihan user sbg PERKALIAN: hasil = tekstur *
    warna. Kalau tekstur rata-rata cuma ~18% terang, warna apa pun yang
    dipilih (termasuk "Racing Red" pekat) ikut teredam jadi ~18% kecerahannya
    -- itu sebabnya user lihat merah yang dipilih keluar sebagai marun nyaris
    hitam. Teksturnya sendiri TIDAK BISA diganti warna solid (perlu detail
    grain/jahitan tetap kelihatan lewat shading), jadi solusinya: regangkan
    (stretch) rentang kecerahan yang ADA ke pita yang jauh lebih terang &
    sempit (lihat BRIGHTEN_*), supaya pola shading relatif (alur jahitan
    lebih gelap, puncak lebih terang) tetap kebaca tapi dasarnya dekat putih
    -- hasil kali dengan warna pilihan jadi mendekati warna aslinya, bukan
    teredam parah."""
    lo = np.percentile(arr, BRIGHTEN_LO_PCT)
    hi = np.percentile(arr, BRIGHTEN_HI_PCT)
    stretched = (arr - lo) / max(hi - lo, 1e-6)
    out = BRIGHTEN_TARGET_LO + stretched * (BRIGHTEN_TARGET_HI - BRIGHTEN_TARGET_LO)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    print("Load mesh...")
    scene = trimesh.load(SRC_GLB, process=False)
    name = list(scene.geometry.keys())[0]
    mesh = scene.geometry[name]

    print("Subdivide segitiga besar (biar batas pattern/main tidak blocky)...")
    # Klasifikasi di bawah ini kerjanya SATU label per segitiga (sample titik
    # tengah segitiga, lalu seluruh segitiga ikut label itu) -- cocok kalau
    # semua segitiga kecil & seragam, tapi mesh hasil Meshy AI ini sangat TIDAK
    # merata: sebagian besar segitiga kecil, tapi ada ekor segitiga RAKSASA
    # (99th percentile ~10x median edge length, lihat diagnosa render flat-
    # color debug -- batas pattern/main yang "robek kertas" ternyata BUKAN
    # noise klasifikasi, tapi persis mengikuti tepi segitiga BESAR itu). Pecah
    # dulu segitiga yang lebih besar dari MAX_EDGE_SUBDIV supaya resolusi
    # klasifikasi cukup halus mengikuti kontur mask 2D yang sudah mulus.
    mesh = mesh.subdivide_to_size(max_edge=MAX_EDGE_SUBDIV, max_iter=12)
    print(f"  segitiga setelah subdivide: {len(mesh.faces)}")

    uv = mesh.visual.uv
    img = mesh.visual.material.baseColorTexture.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    H, W = arr.shape[0], arr.shape[1]

    print("Terangkan tekstur dasar (biar tint warna customizer tidak gelap)...")
    tint_arr = brighten_for_tint(arr)
    tint_img = Image.fromarray(tint_arr, mode="RGB")
    print(f"  mean sebelum: {arr.mean(axis=(0,1))}  sesudah: {tint_arr.astype(np.float32).mean(axis=(0,1))}")

    print("Hitung warna rata-rata tiap segitiga (buat deteksi list aksen)...")
    face_uv = uv[mesh.faces].mean(axis=1)
    px_uv = np.clip((face_uv[:, 0] * W).astype(int), 0, W - 1)
    py_uv = np.clip(((1 - face_uv[:, 1]) * H).astype(int), 0, H - 1)
    face_rgb = arr[py_uv, px_uv, :] / 255.0
    hue, sat, val = rgb_to_hsv_np(face_rgb)
    green = (hue >= 100) & (hue <= 200) & (sat > 0.1) & (val > 0.1)

    n = len(mesh.faces)
    print(f"  accent (hijau/teal): {green.sum()} ({green.sum()/n*100:.1f}%)")

    print("Bangun mask area panel berlian dari render depan...")
    pattern_mask_2d, obj_mask_2d = build_pattern_mask_from_render()

    print("Load info kamera & proyeksikan tiap segitiga (toleransi kedalaman)...")
    with open(CAM_JSON) as f:
        cam = json.load(f)
    px, py, visible = project_with_depth_tolerance(mesh, cam)
    print(f"  segitiga di dekat permukaan depan: {visible.sum()} ({visible.sum()/n*100:.1f}%)")

    in_pattern_2d = np.zeros(n, dtype=bool)
    valid = visible
    in_pattern_2d[valid] = pattern_mask_2d[py[valid], px[valid]]

    labels = np.full(n, "leather_main", dtype=object)
    labels[in_pattern_2d] = "leather_pattern"
    labels[green] = "leather_accent"  # warna prioritas di atas motif

    # CATATAN (sempat dicoba, DIBATALKAN): segitiga yg gagal tes "near_front"
    # pernah diwarisi labelnya dari tetangga 3D terdekat yg lolos tes, dgn
    # asumsi mereka cuma lekukan relief di permukaan DEPAN yg sama. Ternyata
    # itu BUKAN sumber "robek kertas" di depan (akar masalahnya ada di
    # resolusi segitiga -- lihat SUBDIVIDE di atas), dan warisan tetangga-
    # terdekat ini malah BOCOR ke permukaan BAWAH/BELAKANG dudukan jok (shell
    # tipis, jadi tetangga 3D terdekat dari sisi bawah bisa jadi permukaan
    # depan yg searah normal tapi beda sisi) -- kelihatan kalau diputar lihat
    # dari belakang/bawah. Default polos ke "leather_main" utk segitiga yg
    # gagal tes (seperti semula) ternyata lebih aman & hasil depannya tetap
    # bersih berkat subdivide di atas.

    for lbl in np.unique(labels):
        c = (labels == lbl).sum()
        print(f"  {lbl}: {c} segitiga ({c/n*100:.1f}%)")

    print("Bersihkan noise via voting mayoritas spasial (KD-tree)...")
    # CATATAN: dulu pakai connected-components di atas mesh.face_adjacency
    # (gabungkan island kecil ke mayoritas tetangga topologis) -- ternyata
    # itu JUSTRU SUMBER noise "garam-merica" utama, bukan solusinya. Model
    # ini hasil scan/decimation yang mesh-nya sendiri sudah terpecah ~2000
    # potongan kecil TERPUTUS (tidak nyambung lewat shared edge) walau
    # posisinya berdempetan rapat secara spasial. Akibatnya banyak kelompok
    # segitiga yang SUDAH BENAR klasifikasinya (misal bagian tengah panel
    # berlian) kebetulan jadi island kecil secara topologi mesh (<MIN_ISLAND
    # segitiga tersambung), lalu di-vote ikut tetangga topologis yang
    # berseberangan label-nya (nyambung lewat seam/retakan mesh ke area
    # lain) -- hasilnya belang acak walau label awal (sebelum cleanup) sudah
    # rapi (sudah diverifikasi lewat render debug tanpa langkah cleanup ini).
    #
    # Fix: voting mayoritas pakai jarak SPASIAL (KD-tree centroid segitiga),
    # bukan topologi mesh -- jadi kelompok yang padat & benar secara spasial
    # tetap dipertahankan walau mesh-nya retak, sementara noise asli (segitiga
    # nyasar yang tetangga spasialnya mayoritas beda label) tetap dibersihkan.
    # PENTING: KD-tree jarak Euclidean MENTAH bisa salah pilih tetangga --
    # model ini permukaan tipis (shell), jadi titik di sisi LUAR (misal
    # puncak motif berlian yg kelihatan) & titik di sisi DALAM yg cuma
    # beberapa mm di baliknya (menghadap arah berlawanan) sering kali lebih
    # "dekat" secara jarak lurus daripada tetangga di permukaan yg sama.
    # Makanya query tetangga agak banyak (k besar) lalu DIFILTER cuma yang
    # normalnya mirip (dot > 0) sebelum voting -- supaya tidak numpang
    # vote dari sisi lain shell yang jelas tidak relevan.
    # CATATAN TAMBAHAN: voting mayoritas PENUH (ambil label terbanyak begitu
    # saja, walau cuma menang tipis) ternyata JUGA menghasilkan noise baru --
    # label berubah di jumlah segitiga yang kecil secara total (~1%), tapi
    # tersebar pervasif (banyak kemenangan tipis/hampir seri di sekitar
    # batas motif berlian yang memang rame & tidak stabil), jadi hasilnya
    # malah lebih berbintik dibanding tanpa cleanup sama sekali. Makanya
    # dipakai ambang KONSERVATIF -- cuma balik label kalau mayoritas
    # tetangga spasial BENAR-BENAR dominan (>=85%) ke label lain; kalau
    # tidak, biarkan label asli (klasifikasi dari proyeksi mask 2D, yang
    # sudah terbukti rapi) apa adanya.
    # Satu kali voting dgn ambang 85% masih nyisain cukup banyak noise kecil
    # (kelihatan di live site) -- diulang beberapa KALI dgn ambang lebih
    # longgar (lihat SPATIAL_VOTE_DOMINANCE) supaya noise terkikis bertahap
    # tiap putaran, tanpa jatuh ke masalah "voting penuh" (ping-pong/checkerboard)
    # krn tiap putaran tetap butuh mayoritas yang cukup dominan.
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    tree = cKDTree(centroids)
    _, neighbor_idx = tree.query(centroids, k=SPATIAL_VOTE_K)
    same_side = np.einsum("ij,ikj->ik", normals, normals[neighbor_idx]) > 0
    clean = labels.copy()
    for p in range(SPATIAL_VOTE_PASSES):
        neighbor_labels = clean[neighbor_idx]  # (n, k)
        new_clean = clean.copy()
        changed = 0
        for i in range(n):
            filt = neighbor_labels[i][same_side[i]]
            if len(filt) < 6:
                continue
            vals, counts = np.unique(filt, return_counts=True)
            top = np.argmax(counts)
            if vals[top] != clean[i] and counts[top] / len(filt) >= SPATIAL_VOTE_DOMINANCE:
                new_clean[i] = vals[top]
                changed += 1
        clean = new_clean
        print(f"  putaran {p+1}: {changed} segitiga berubah label")
        if changed == 0:
            break
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
        # pakai object TEXTURE yang sama persis (bukan copy()/deep-copy) di
        # ketiga material supaya exporter glTF simpan 1 salinan gambar saja
        # (bukan 3x) -- tint warna tetap jalan per-material lewat THREE.js
        # material.color di runtime, tidak butuh baseColorTexture terpisah.
        # PENTING: baseColorTexture di GLB ini HARUS tekstur ASLI (gelap),
        # bukan tint_img -- supaya tampilan "warna asli" (sebelum user pilih
        # warna apa pun) tetap sesuai foto produk, bukan pucat/putih. tint_img
        # (versi diterangkan) diekspor TERPISAH (lihat TINT_BASE_PNG) dan
        # cuma dipakai dummy-kursi-twinseat.html sbg basis tekstur SAAT user
        # memilih warna custom (ditukar via material.map saat itu saja) --
        # lihat penjelasan brighten_for_tint kenapa butuh 2 versi tekstur.
        cluster_mat = trimesh.visual.material.PBRMaterial(
            name=cluster,
            baseColorTexture=src_mat.baseColorTexture,
            baseColorFactor=src_mat.baseColorFactor,
            metallicFactor=src_mat.metallicFactor,
            roughnessFactor=src_mat.roughnessFactor,
            metallicRoughnessTexture=src_mat.metallicRoughnessTexture,
            normalTexture=src_mat.normalTexture,
            emissiveFactor=src_mat.emissiveFactor,
            emissiveTexture=src_mat.emissiveTexture,
            doubleSided=src_mat.doubleSided,
        )
        sub.visual = trimesh.visual.TextureVisuals(uv=uv, material=cluster_mat)
        sub.remove_unreferenced_vertices()
        out_scene.add_geometry(sub, node_name=cluster, geom_name=cluster)

    out_scene.export(OUT_GLB, include_normals=True)
    print("Selesai ->", OUT_GLB, f"{os.path.getsize(OUT_GLB)/1e6:.1f} MB")

    tint_img.save(TINT_BASE_PNG)
    print("Selesai ->", TINT_BASE_PNG)


if __name__ == "__main__":
    main()
