# Pasang rangka dari scan baru (Mesh_0.001) menggantikan rangka lama di objek Kursi. Hasil disimpan ke blend baru (blend asli tidak diubah).
import bpy, bmesh, numpy as np, sys, math
SRC,OUT=sys.argv[sys.argv.index('--')+1:][:2]
ZC=-0.62          # potong: hanya bagian di bawah z ini (kaki, alas, palang)
MINF=1         # buang serpihan < sekian face
XL=-2.52          # buang sisa cangkang jok di kiri scan (koordinat dunia)
TARGET=85000      # jumlah face rangka setelah di-decimate
bpy.ops.wm.open_mainfile(filepath=SRC)
def conn(nv,e):
    lab=np.arange(nv)
    while True:
        m=np.minimum(lab[e[:,0]],lab[e[:,1]]); new=lab.copy(); np.minimum.at(new,e[:,0],m); np.minimum.at(new,e[:,1],m); new=new[new]
        if (new==lab).all(): return lab
        lab=new
def tris(d):
    lt=np.empty(len(d.polygons),np.int32); d.polygons.foreach_get('loop_total',lt); assert (lt==3).all()
    li=np.empty(len(d.loops),np.int32); d.loops.foreach_get('vertex_index',li); return li.reshape(-1,3)
def world(o):
    d=o.data; v=np.empty(len(d.vertices)*3); d.vertices.foreach_get('co',v); v=v.reshape(-1,3); M=np.array(o.matrix_world); return v@M[:3,:3].T+M[:3,3]
# ---- 1. ambil rangka dari scan
sc_o=bpy.data.objects['Mesh_0.001']; V=world(sc_o); F=tris(sc_o.data)
sel=(V[F][:,:,2]<ZC).all(1)&(V[F][:,:,0]>XL).all(1); F1=F[sel]
vs=np.unique(F1); mp=-np.ones(len(V),np.int64); mp[vs]=np.arange(len(vs)); F1=mp[F1]; V1=V[vs]
e=np.concatenate([F1[:,[0,1]],F1[:,[1,2]]]); lab=conn(len(V1),e); fl=lab[F1[:,0]]
u,inv,cnt=np.unique(fl,return_inverse=True,return_counts=True); keep=cnt[inv]>=MINF
F2=F1[keep]; vs2=np.unique(F2); mp2=-np.ones(len(V1),np.int64); mp2[vs2]=np.arange(len(vs2)); F2=mp2[F2]; V2=V1[vs2]
print('rangka baru: face',len(F1),'->',len(F2),'komponen',int((cnt>=MINF).sum()),'dari',len(cnt),'ukuran',sorted(cnt)[::-1][:15])
# ---- 2. posisi: cocokkan alas (z<-0.9) rangka lama
K=bpy.data.objects['Kursi']; Vk=world(K); Fk=tris(K.data)
ing=np.array([len(x.groups)>0 for x in K.data.vertices])
fz=Vk[Fk][:,:,2]; selk=(fz<ZC).all(1)&(~ing[Fk]).all(1); idx=np.where(selk)[0]
vk=np.unique(Fk[idx]); mk=-np.ones(len(Vk),np.int64); mk[vk]=np.arange(len(vk)); ek=np.concatenate([mk[Fk[idx][:,[0,1]]],mk[Fk[idx][:,[1,2]]]])
lk=conn(len(vk),ek); fk=lk[mk[Fk[idx][:,0]]]; u2,inv2,c2=np.unique(fk,return_inverse=True,return_counts=True)
old_frame=idx; print('rangka lama: face',len(old_frame))
ov=Vk[np.unique(Fk[old_frame])]
def base(P): m=P[:,2]<-0.9; q=P[m]; return q.min(0),q.max(0)
a0,a1=base(ov); b0,b1=base(V2); print('alas lama',a0.round(3),a1.round(3),'baru',b0.round(3),b1.round(3))
T=np.array([((a0+a1)/2-(b0+b1)/2)[0],((a0[1]+a1[1])/2-(b0[1]+b1[1])/2),a0[2]-b0[2]])
print('geser',T.round(3)); V2=V2+T
# ---- 3. buang rangka lama dari Kursi (lubang di bawah jok dibiarkan terbuka; tertutup kaki rangka baru)
bm=bmesh.new(); bm.from_mesh(K.data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm,geom=[bm.faces[i] for i in old_frame],context='FACES')
bmesh.ops.delete(bm,geom=[x for x in bm.verts if not x.link_faces],context='VERTS')
bm.to_mesh(K.data); bm.free()
# ---- 4. objek rangka (decimate + shade smooth + material logam gelap)
me=bpy.data.meshes.new('Rangka'); me.from_pydata(V2.tolist(),[],F2.tolist()); me.update()
ob=bpy.data.objects.new('Rangka',me); bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active=ob
VOX=float(__import__('os').environ.get('VOX','0.012'))
so=ob.modifiers.new('sol','SOLIDIFY'); so.thickness=float(__import__('os').environ.get('THK','0.02')); so.offset=0; so.use_even_offset=True
rm=ob.modifiers.new('rem','REMESH'); rm.mode='VOXEL'; rm.voxel_size=VOX; rm.use_smooth_shade=True
bpy.ops.object.modifier_apply(modifier='sol'); bpy.ops.object.modifier_apply(modifier='rem')
# buang pulau kecil hasil remesh
bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
Fm=np.array([[v.index for v in f.verts] for f in bm.faces]); lb=conn(len(bm.verts),np.concatenate([Fm[:,[0,1]],Fm[:,[1,2]]])); uu,ii,cc=np.unique(lb[Fm[:,0]],return_inverse=True,return_counts=True)
bmesh.ops.delete(bm,geom=[bm.faces[i] for i in np.where(cc[ii]<int(__import__('os').environ.get('ISL','1500')))[0]],context='FACES')
bmesh.ops.delete(bm,geom=[x for x in bm.verts if not x.link_faces],context='VERTS'); bm.to_mesh(me); bm.free()
print('setelah remesh face',len(me.polygons))
md=ob.modifiers.new('dec','DECIMATE'); md.ratio=min(1.0,TARGET/max(len(me.polygons),1))
bpy.ops.object.modifier_apply(modifier='dec')
for p in me.polygons: p.use_smooth=True
mat=bpy.data.materials.new('rangka_logam'); mat.use_nodes=True
b=mat.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.30,0.31,0.33,1); b.inputs['Roughness'].default_value=0.5; b.inputs['Metallic'].default_value=0.5
me.materials.append(mat)
print('rangka final face',len(me.polygons))
sc_o.hide_viewport=True; sc_o.hide_render=True
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('OB saved',OUT)
