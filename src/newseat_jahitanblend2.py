# .blend untuk edit jahitan di KEDUA kursi: label Tali (rapi + salinan kiri), warna viewport kontras.
import bpy, numpy as np, sys
SRC,LAB,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')   # file disimpan saat Edit Mode: keluar dulu agar perubahan tersimpan
me=o.data
COL={'1_KulitUtama_hitam':(0.10,0.10,0.11),'2_MotifBerlian_panel':(0.10,0.55,0.20),'3_ListAksen_hijau':(0.10,0.75,0.75),'4_TidakBisaDiganti':(0.55,0.55,0.60),'Tali':(1.0,0.45,0.0)}
IDX={'1_KulitUtama_hitam':0,'2_MotifBerlian_panel':1,'3_ListAksen_hijau':2,'4_TidakBisaDiganti':3,'Tali':4}
slot={}
for i,m in enumerate(me.materials):
    n=m.name.split('.')[0]; slot[IDX[n]]=i
    c=COL[n]; m.diffuse_color=(*c,1)
    if m.use_nodes:
        b=m.node_tree.nodes.get('Principled BSDF')
        if b: b.inputs['Base Color'].default_value=(*c,1)
lab=np.load(LAB).astype(np.int32); assert len(lab)==len(me.polygons)
me.polygons.foreach_set('material_index',np.array([slot[k] for k in range(5)],np.int32)[lab]); me.update()
for scr in bpy.data.screens:
    for a in scr.areas:
        if a.type=='VIEW_3D':
            for sp in a.spaces:
                if sp.type=='VIEW_3D': sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'
mi2=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('material_index',mi2); print('cek',np.bincount(mi2),[m.name for m in me.materials])
bpy.ops.wm.save_as_mainfile(filepath=OUT,compress=True); print('done')
