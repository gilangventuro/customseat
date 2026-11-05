"""
Konversi assets/twin-meshy/model.fbx (hasil Meshy AI, 1 mesh + PBR texture
lengkap: base color, metallic, roughness, normal) menjadi
assets/twin-seat-meshy.glb yang ringan untuk web:
  - decimate ke target triangle count
  - resize & re-encode semua texture ke JPG/PNG lebih kecil
  - export GLB dengan draco compression

Jalankan lewat Blender (bukan python biasa -- butuh bpy):
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python src/convert_twin_meshy.py
"""
import bpy
import os
import math

SRC_FBX = os.path.abspath("assets/twin-meshy/model.fbx")
OUT_GLB = os.path.abspath("assets/twin-seat-meshy.glb")
TARGET_TRIS = 150_000
TEX_MAX_SIZE = 2048

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print("Import FBX...")
    bpy.ops.import_scene.fbx(filepath=SRC_FBX)

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    print(f"Mesh objects: {len(meshes)}")
    for m in meshes:
        tris = sum(len(p.vertices) - 2 for p in m.data.polygons)
        print(f"  {m.name}: {len(m.data.vertices)} verts, ~{tris} tris")

    total_tris = sum(sum(len(p.vertices) - 2 for p in m.data.polygons) for m in meshes)
    print(f"Total triangles: {total_tris}")

    # ---------- decimate tiap mesh proporsional ke target ----------
    if total_tris > TARGET_TRIS:
        ratio = TARGET_TRIS / total_tris
        print(f"Decimate ratio: {ratio:.4f}")
        for m in meshes:
            bpy.context.view_layer.objects.active = m
            mod = m.modifiers.new(name="Decimate", type='DECIMATE')
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
        new_total = sum(sum(len(p.vertices) - 2 for p in m.data.polygons) for m in meshes)
        print(f"Triangles setelah decimate: {new_total}")

    # ---------- resize semua image texture ----------
    print("Resize textures...")
    for img in bpy.data.images:
        if img.size[0] > TEX_MAX_SIZE or img.size[1] > TEX_MAX_SIZE:
            w, h = img.size
            scale = TEX_MAX_SIZE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"  {img.name}: {w}x{h} -> {new_w}x{new_h}")
            img.scale(new_w, new_h)

    # ---------- export GLB ----------
    print("Export GLB...")
    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB,
        export_format='GLB',
        export_texture_dir='',
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_materials='EXPORT',
        export_image_format='AUTO',
    )
    size_mb = os.path.getsize(OUT_GLB) / 1e6
    print(f"Selesai -> {OUT_GLB} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
