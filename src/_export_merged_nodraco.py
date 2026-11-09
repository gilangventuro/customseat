import bpy, os
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath("_tmp_merged.glb"))
bpy.ops.export_scene.gltf(
    filepath=os.path.abspath("_tmp_merged_nodraco.glb"),
    export_format='GLB',
    export_draco_mesh_compression_enable=False,
    export_materials='EXPORT',
    export_image_format='AUTO',
)
print("done")
