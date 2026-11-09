"""
Re-import hasil klasifikasi 3-material (_tmp_classified.glb, tanpa draco)
dan export ulang jadi assets/twin-seat-meshy.glb final dengan draco
compression -- langkah terakhir pipeline split_twin_seats.py (lihat
docstring di sana untuk alur lengkap).

Jalankan lewat Blender:
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python src/_compress_final.py
"""
import bpy
import os

SRC_GLB = os.path.abspath("_tmp_classified.glb")
OUT_GLB = os.path.abspath("assets/twin-seat-meshy.glb")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Import GLB...")
    bpy.ops.import_scene.gltf(filepath=SRC_GLB)

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    print(f"Mesh objects: {len(meshes)}")
    for m in meshes:
        print(f"  {m.name}: materials = {[s.material.name for s in m.material_slots]}")

    print("Dedup texture gambar (3 material asalnya dari 1 tekstur yang sama)...")
    # importer bikin Image datablock baru per material walau isinya identik
    # byte-per-byte -- paksa semua material pakai Image datablock milik
    # material pertama (urutan Image Texture node sama persis di ketiganya),
    # baru hapus datablock yang sudah tak terpakai supaya exporter glTF
    # cuma nulis 1 salinan tiap tekstur (bukan 3x).
    mats = [m for m in bpy.data.materials if m.name in ("leather_main", "leather_pattern", "leather_accent")]
    ref_mat = next(m for m in mats if m.name == "leather_main")
    ref_images = [n.image for n in ref_mat.node_tree.nodes if n.type == 'TEX_IMAGE']
    for mat in mats:
        if mat is ref_mat:
            continue
        tex_nodes = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE']
        for n, ref_img in zip(tex_nodes, ref_images):
            n.image = ref_img
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)
    print("  images tersisa:", [img.name for img in bpy.data.images])

    print("Export GLB (draco)...")
    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB,
        export_format='GLB',
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_materials='EXPORT',
        export_image_format='AUTO',
    )
    size_mb = os.path.getsize(OUT_GLB) / 1e6
    print(f"Selesai -> {OUT_GLB} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
