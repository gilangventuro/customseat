import bpy, numpy as np, os, sys
out=os.path.abspath(sys.argv[sys.argv.index('--')+1])
lab=np.load('_tmp_lab.npy')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="assets/twin-seat-meshy.glb")
ms=[o for o in bpy.context.scene.objects if o.type=='MESH']
byname={}
for o in ms:
    for s in o.material_slots: byname[s.material.name]=s.material
rest=byname['leather_main'].copy(); rest.name='leather_rest'
order=[byname['leather_main'],byname['leather_pattern'],byname['leather_accent'],rest]
refimgs=[n.image for n in byname['leather_main'].node_tree.nodes if n.type=='TEX_IMAGE']
for mat in order[1:]:
    for n,r in zip([n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE'],refimgs): n.image=r
off=0
for o in ms:
    me=o.data; k=len(me.polygons)
    me.materials.clear()
    for mat in order: me.materials.append(mat)
    me.polygons.foreach_set('material_index',lab[off:off+k].astype(np.int32)); me.update()
    off+=k
assert off==len(lab)
for img in list(bpy.data.images):
    if img.users==0: bpy.data.images.remove(img)
bpy.ops.export_scene.gltf(filepath=out,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',out,os.path.getsize(out)/1e6)
