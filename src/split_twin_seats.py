"""
Pecah assets/twin-seat-meshy.glb (1 mesh gabungan, 2 kursi captain
bersebelahan) menjadi 2 sub-mesh terpisah -- kursi kiri & kursi kanan --
masing-masing dengan material sendiri (duplikat dari material asli, jadi
teksturnya identik) supaya warnanya bisa diganti independen per sisi lewat
tint di browser.

Pemisahan berdasarkan posisi X pusat tiap segitiga (sumbu kiri-kanan model),
dibagi tepat di tengah bounding box.

Jalankan lewat Blender:
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python src/split_twin_seats.py
"""
import bpy
import os

SRC_GLB = os.path.abspath("assets/twin-seat-meshy.glb")
OUT_GLB = os.path.abspath("assets/twin-seat-meshy.glb")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Import GLB...")
    bpy.ops.import_scene.gltf(filepath=SRC_GLB)

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    assert len(meshes) == 1, f"expected 1 mesh, got {len(meshes)}"
    obj = meshes[0]

    xs = [v.co.x for v in obj.data.vertices]
    mid_x = (min(xs) + max(xs)) / 2
    print(f"split at local X = {mid_x:.5f} (range {min(xs):.5f}..{max(xs):.5f})")

    src_mat = obj.data.materials[0]
    mat_left = src_mat.copy()
    mat_left.name = "leather_left"
    mat_right = src_mat.copy()
    mat_right.name = "leather_right"
    obj.data.materials.append(mat_left)
    obj.data.materials.append(mat_right)
    idx_left = len(obj.data.materials) - 2
    idx_right = len(obj.data.materials) - 1

    mesh = obj.data
    for poly in mesh.polygons:
        cx = sum(mesh.vertices[vi].co.x for vi in poly.vertices) / len(poly.vertices)
        poly.material_index = idx_left if cx < mid_x else idx_right

    # buang slot material lama (index 0) yang sudah tidak dipakai segitiga manapun
    bpy.context.view_layer.objects.active = obj
    obj.active_material_index = 0
    bpy.ops.object.material_slot_remove_unused()

    print("materials setelah split:", [m.name for m in obj.data.materials])
    left_tris = sum(1 for p in mesh.polygons if obj.data.materials[p.material_index].name == "leather_left")
    right_tris = sum(1 for p in mesh.polygons if obj.data.materials[p.material_index].name == "leather_right")
    print(f"kiri: {left_tris} segitiga, kanan: {right_tris} segitiga")

    print("Export GLB...")
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
