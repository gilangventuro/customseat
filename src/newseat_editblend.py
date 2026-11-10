# Bikin .blend siap edit manual: 4 material per-label (warna beda di viewport), tekstur ter-pack.
import bpy, numpy as np, sys, os
BLEND,LAB,BASE,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]; o.name='Kursi'
lab=np.load(LAB).astype(np.int32); me=o.data; me.materials.clear()
img=bpy.data.images.load(BASE)
cols={'1_KulitUtama_hitam':(0.12,0.12,0.12),'2_MotifBerlian_panel':(0.9,0.5,0.1),'3_ListAksen_hijau':(0.1,0.8,0.7),'4_TidakBisaDiganti':(0.9,0.1,0.1)}
for n,c in cols.items():
    m=bpy.data.materials.new(n); m.use_nodes=True; m.diffuse_color=(*c,1)
    bs=m.node_tree.nodes['Principled BSDF']; bs.inputs['Base Color'].default_value=(*c,1)
    me.materials.append(m)
me.polygons.foreach_set('material_index',lab); me.update()
img.pack()
for area in bpy.context.screen.areas if bpy.context.screen else []:
    pass
for scr in bpy.data.screens:
    for a in scr.areas:
        if a.type=='VIEW_3D':
            for sp in a.spaces:
                if sp.type=='VIEW_3D': sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'
bpy.ops.wm.save_as_mainfile(filepath=OUT,compress=True)
print('done',os.path.getsize(OUT)/1e6)
