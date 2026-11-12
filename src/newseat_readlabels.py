# Baca label per-face dari .blend editan user, dipetakan berdasarkan NAMA material (bukan nomor slot).
import bpy, numpy as np, sys
BLEND,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]; me=o.data
MAP={'1_KulitUtama_hitam':0,'2_MotifBerlian_panel':1,'3_ListAksen_hijau':2,'4_TidakBisaDiganti':3,'Tali':4}
names=[m.name if m else None for m in me.materials]; print('slots',names,'faces',len(me.polygons))
mi=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('material_index',mi)
lut=np.array([MAP.get((n or '').split('.')[0],-1) for n in names]+[-1])
lab=lut[np.minimum(mi,len(names))].astype(np.int8); assert (lab>=0).all(),'material tak dikenal'
np.save(OUT,lab); print('count',np.bincount(lab))
