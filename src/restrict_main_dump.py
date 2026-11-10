import bpy, numpy as np, bmesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="assets/twin-seat-meshy.glb")
ms=[o for o in bpy.context.scene.objects if o.type=='MESH']
print([(o.name,[s.material.name for s in o.material_slots],len(o.data.polygons)) for o in ms])
C=[];N=[];M=[];A=[]
for o in ms:
    me=o.data; mw=o.matrix_world
    names=[s.material.name for s in o.material_slots]
    me.calc_loop_triangles()
    for p in me.polygons:
        pass
    n=len(me.polygons)
    cen=np.empty(n*3);nor=np.empty(n*3);mi=np.empty(n,dtype=np.int32);ar=np.empty(n)
    me.polygons.foreach_get('center',cen);me.polygons.foreach_get('normal',nor);me.polygons.foreach_get('material_index',mi);me.polygons.foreach_get('area',ar)
    cen=cen.reshape(-1,3);nor=nor.reshape(-1,3)
    R=np.array(mw.to_3x3());T=np.array(mw.translation)
    C.append(cen@R.T+T);N.append(nor@R.T);M.append(np.array([names[i] for i in mi]));A.append(ar)
np.savez('_tmp_faces.npz',c=np.concatenate(C),n=np.concatenate(N),m=np.concatenate(M),a=np.concatenate(A))
