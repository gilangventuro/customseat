SIMULATOR WARNA & POLA JAHITAN JOK — ALLDILA SEATS
===================================================

CARA PAKAI (untuk presentasi ke klien)
--------------------------------------
1. Buka file  simulator-warna-jok.html  (klik dua kali, terbuka di Chrome/Edge/Safari).
2. Tidak perlu internet. Foto & semua data sudah ada di dalam file itu.
3. Untuk dikirim ke klien, cukup kirim file  simulator-warna-jok.html  saja.

Fitur:
- Ganti warna 8 elemen: Kulit Utama, Panel Tengah, Panel Samping, Aksen/Strip,
  Benang Jahitan, Cangkang Armrest, Trim Armrest, Bordir Logo.
- Pola Jahitan (quilting) di panel tengah: Diamond, Kotak, Honeycomb,
  Garis Horizontal, Garis Vertikal, Chevron — model Single/Double, ukuran Kecil/Sedang/Besar.
- Klik langsung bagian kursi di foto untuk memilih elemen.
- "Tahan: lihat asli" untuk membandingkan dengan foto asli.
- "Latar studio" untuk latar bersih saat presentasi.
- Preset warna siap pakai.
- "Unduh PNG" = foto hasil + lembar spesifikasi warna & pola (bisa diisi nama klien/proyek).

Catatan: ini simulasi visual; warna aktual material bisa sedikit berbeda.


ISI FOLDER
----------
simulator-warna-jok.html   File utama (siap pakai, mandiri).
contoh-hasil.jpg           Contoh hasil ganti warna.
contoh-pola-jahitan.jpg    Contoh 6 pola jahitan.
assets/foto-asli.webp      Foto asli kursi.
src/                       Sumber untuk mengedit simulator (tidak perlu untuk klien):
  template.html            Tampilan & logika simulator (palet warna, preset, pola).
  build.py                 Menggabungkan template + foto + mask menjadi file utama.
  masks/                   Peta area tiap elemen hasil olah foto.


MENGEDIT (untuk tim)
--------------------
- Ubah palet warna / nama warna / preset: edit bagian LEATHER, THREAD, SHELL, TRIM,
  LOGO dan PRESETS di  src/template.html.
- Setelah mengedit, jalankan (butuh Python 3):
      python3 src/build.py
  File  simulator-warna-jok.html  akan dibuat ulang.
- Mask di src/masks dibuat khusus untuk foto ini. Foto/model kursi lain perlu
  dibuatkan mask baru.
