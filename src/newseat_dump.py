# Langkah A (Blender): impor FBX captain seat baru, decimate ke ~500k face, simpan .blend + dump npz per-face.
import bpy, bmesh, numpy as np, os, sys
FBX=sys.argv[sys.argv.index('--')+1]; OUT_BLEND=sys.argv[sys.argv.index('--')+2]; OUT_NPZ=sys.argv[sys.argv.index('--')+3]
RATIO=float(sys.argv[sys.argv.index('--')+4]) if len(sys.argv)>sys.argv.index('--')+4 else 0.25
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
# terapkan transform (rotasi -90 X dari FBX) supaya koordinat dump = dunia
bpy.ops.object.select_all(action='DESELECT'); o.select_set(True)
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
m=o.modifiers.new('d','DECIMATE'); m.decimate_type='COLLAPSE'; m.ratio=RATIO; m.use_collapse_triangulate=True
bpy.ops.object.modifier_apply(modifier='d')
me=o.data; me.calc_loop_triangles()
n=len(me.polygons); print('faces',n)
co=np.empty(len(me.vertices)*3); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
pv=np.empty(n*3,np.int32); me.polygons.foreach_get('vertices',pv); pv=pv.reshape(n,3)
nor=np.empty(n*3); me.polygons.foreach_get('normal',nor); nor=nor.reshape(n,3)
uvl=me.uv_layers.active.data; uv=np.empty(len(uvl)*2); uvl.foreach_get('uv',uv); uv=uv.reshape(-1,3-1)
uvf=uv.reshape(n,3,2).mean(1)
cen=co[pv].mean(1)
np.savez(OUT_NPZ,pv=pv,co=co,nor=nor,uv=uvf,cen=cen)
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
