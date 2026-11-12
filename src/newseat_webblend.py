# .blend versi web saat ini: label material akhir (tanpa Tali) + objek benang jahitan 3D terpisah ("Jahitan").
import bpy, numpy as np, sys
SRC,LAB,THR,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')   # file disimpan saat Edit Mode: keluar dulu agar perubahan tersimpan
me=o.data
slot={}
for i,m in enumerate(me.materials):
    n=(m.name if m else '').split('.')[0]; slot[{'1_KulitUtama_hitam':0,'2_MotifBerlian_panel':1,'3_ListAksen_hijau':2,'4_TidakBisaDiganti':3,'Tali':4}[n]]=i
lab=np.load(LAB).astype(np.int32); assert len(lab)==len(me.polygons)
me.polygons.foreach_set('material_index',np.array([slot[k] for k in range(5)],np.int32)[lab]); me.update()
z=np.load(THR); tm=bpy.data.meshes.new('JahitanMesh'); tm.from_pydata(z['V'].tolist(),[],z['F'].tolist()); tm.update()
to=bpy.data.objects.new('Jahitan',tm); bpy.context.scene.collection.objects.link(to)
sm=bpy.data.materials.new('Benang_Jahitan'); sm.use_nodes=True; sm.diffuse_color=(0.9,0.45,0.1,1)
b=sm.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.9,0.45,0.1,1); b.inputs['Roughness'].default_value=0.55
tm.materials.append(sm)
for pl in tm.polygons: pl.use_smooth=True
bpy.ops.wm.save_as_mainfile(filepath=OUT,compress=True); print('done')
