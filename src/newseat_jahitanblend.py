# Buat salinan .blend dari editan user dengan label jahitan (Tali) hasil rapikan + salinan ke kursi sebelah.
import bpy, numpy as np, sys
SRC,LAB,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]; me=o.data
slot={}
for i,m in enumerate(me.materials):
    n=(m.name if m else '').split('.')[0]; slot[{'1_KulitUtama_hitam':0,'2_MotifBerlian_panel':1,'3_ListAksen_hijau':2,'4_TidakBisaDiganti':3,'Tali':4}[n]]=i
lab=np.load(LAB).astype(np.int32); assert len(lab)==len(me.polygons)
mi=np.array([slot[k] for k in range(5)],np.int32)[lab]
me.polygons.foreach_set('material_index',mi); me.update()
bpy.ops.wm.save_as_mainfile(filepath=OUT,compress=True); print('done')
