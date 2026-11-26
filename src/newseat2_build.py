# Bangun GLB situs dari "SEATS ALLDILA UPDATE 2.blend" klaster 2 (seat Meshy baru: Mesh_0 / Mesh_0.001 + semua aksesori sudah dipasang user).
# Jalankan: blender --background --python src/newseat2_build.py -- <out.glb>   (butuh _tmp_tex/* dari newseat2_classmap.py & newseat2_tex.py)
import bpy,bmesh,numpy as np,json,math,sys,re,os
P='C:/Users/LENOVO LOQ/Projects/alldila-jok-simulator/'
OUT=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else P+'_tmp_new2.glb'
DX=-3.51; DEC=float(os.environ.get('DEC','0.5'))
def log(*a): print(*a,flush=True)
bpy.ops.wm.open_mainfile(filepath=P+'SEATS ALLDILA UPDATE 2.blend')
names={o.name for o in bpy.data.objects if o.type=='MESH'}
for o in bpy.data.objects:
    if o.type=='MESH':
        bpy.context.view_layer.objects.active=o
        if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
# ---- peta nama klaster 2 -> nama kanonik
canon={}
for n in names:
    if n.endswith('.001') and n[:-4] in names and not n.startswith(('Jaring','Kubus','Mesh_0')): canon[n]=n[:-4]
canon['Sandaran_Belakang_Kiri_02.002']='Sandaran_Belakang_Kanan_02'
canon['Handle_Kiri_00.002']='Handle_Kanan_00'
canon['Cupholder_Kiri_00.002']='Cupholder_Kanan_00'
canon['Mesh_0']='Kursi_Kiri'; canon['Mesh_0.001']='Kursi_Kanan'
canon={k:v for k,v in canon.items() if k in names}
log('klaster2',len(canon))
# bahan kanonik dari klaster 1 (per objek, per slot)
src_mats={}
for k,v in canon.items():
    s=bpy.data.objects.get(v) if v not in ('Kursi_Kiri','Kursi_Kanan') else None
    if s is not None and s.name not in canon: src_mats[k]=[sl.material for sl in s.material_slots]
def W(o):
    M=np.array(o.matrix_world); v=np.zeros(len(o.data.vertices)*3); o.data.vertices.foreach_get('co',v); return v.reshape(-1,3)@M[:3,:3].T+M[:3,3]
def setW(o,w):
    o.parent=None; o.matrix_world.identity()
    o.data.vertices.foreach_set('co',w.ravel()); o.data.update()
keep=[]
for k,v in canon.items():
    o=bpy.data.objects[k]; w=W(o); w[:,0]+=DX; setW(o,w); keep.append(o)
_ks={o.name for o in keep}
for o in list(bpy.data.objects):
    if o.name not in _ks: bpy.data.objects.remove(o,do_unlink=True)   # buang klaster 1 & sampah dulu supaya nama kanonik tidak bertabrakan
# ---- bahan kulit bertekstur baru
for nm in ('leather_main','leather_pattern','leather_accent','leather_stitch'):
    if nm in bpy.data.materials: bpy.data.materials[nm].name='old_'+nm
tint=bpy.data.images.load(P+'_tmp_tex/tint2048.jpg'); tint.colorspace_settings.name='sRGB'
nrm=bpy.data.images.load(P+'_tmp_tex/normal2048.jpg'); nrm.colorspace_settings.name='Non-Color'
def leather(nm):
    m=bpy.data.materials.new(nm); m.use_nodes=True; nt=m.node_tree
    b=[n for n in nt.nodes if n.type=='BSDF_PRINCIPLED'][0]
    t=nt.nodes.new('ShaderNodeTexImage'); t.image=tint
    n2=nt.nodes.new('ShaderNodeTexImage'); n2.image=nrm
    nm_=nt.nodes.new('ShaderNodeNormalMap')
    nt.links.new(t.outputs['Color'],b.inputs['Base Color']); nt.links.new(n2.outputs['Color'],nm_.inputs['Color']); nt.links.new(nm_.outputs['Normal'],b.inputs['Normal'])
    b.inputs['Metallic'].default_value=0; b.inputs['Roughness'].default_value=0.55
    m.diffuse_color=(0.2,0.2,0.2,1); return m
LM={nm:leather(nm) for nm in ('leather_main','leather_pattern','leather_accent')}
CL=np.load(P+'_tmp_tex/classmap.npy'); H=CL.shape[0]; FL=json.load(open(P+'_tmp_tex/flatuv.json'))
# ---- badan kursi baru: decimate + kelas per-face dari peta kelas tekstur
for k in ('Mesh_0','Mesh_0.001'):
    o=bpy.data.objects[k]; nn=canon[k]; o.name=nn; o.data.name=nn
    bpy.context.view_layer.objects.active=o
    n0=len(o.data.polygons)
    md=o.modifiers.new('dec','DECIMATE'); md.ratio=DEC; md.delimit={'UV'}
    bpy.ops.object.modifier_apply(modifier='dec')
    me=o.data
    while len(me.color_attributes): me.color_attributes.remove(me.color_attributes[0])
    uv=me.uv_layers.active.data; ua=np.zeros(len(uv)*2); uv.foreach_get('uv',ua); ua=ua.reshape(-1,2)
    cnt=np.zeros(len(me.polygons),dtype=np.int32); me.polygons.foreach_get('loop_total',cnt)
    st=np.concatenate([[0],np.cumsum(cnt)[:-1]]); cen=np.add.reduceat(ua,st,axis=0)/cnt[:,None]
    votes=np.zeros((len(cnt),3),dtype=np.int32)
    def samp(p):
        x=np.clip((p[:,0]%1.0)*(H-1),0,H-1).astype(int); y=np.clip((1-(p[:,1]%1.0))*(H-1),0,H-1).astype(int); return CL[y,x]
    cls=samp(cen); votes[np.arange(len(cnt)),cls]+=1
    for j in range(3):
        idx=np.minimum(st+j,len(ua)-1); pj=ua[idx]*0.7+cen*0.3; c=samp(pj)
        ok=cnt>j; votes[np.arange(len(cnt))[ok],c[ok]]+=1
    fc=votes.argmax(1).astype(np.int32)
    me.materials.clear()
    for nm in ('leather_main','leather_pattern','leather_accent'): me.materials.append(LM[nm])
    me.polygons.foreach_set('material_index',fc)
    for p in me.polygons: p.use_smooth=True
    log('seat',nn,n0,'->',len(me.polygons),np.bincount(fc,minlength=3).tolist())
# ---- bahan objek lain: ikut klaster 1 (kanonik); bantalan legrest -> kulit baru + UV tambalan datar
def strip(n): return re.sub(r'\.\d{3}$','',n)
for o in keep:
    if o.name.startswith('Kursi_'): continue
    nn=canon[o.name]
    old=[sl.material for sl in o.material_slots]; sm=src_mats.get(o.name)
    newm=[]
    for i,m in enumerate(old):
        t=sm[i] if (sm and len(sm)==len(old)) else bpy.data.materials.get(strip(m.name) if m else '', m)
        if t is not None and t.name.startswith('old_'): t=LM[t.name[4:]]
        if t is not None and t.name in LM: t=LM[t.name]
        newm.append(t or m)
    for i,t in enumerate(newm): o.data.materials[i]=t   # jangan clear(): indeks material face ikut ter-reset
    o.name=nn; o.data.name=nn
for nn in ('Footrest_Kiri_Bantalan','Footrest_Kanan_Bantalan'):
    o=bpy.data.objects[nn]; me=o.data
    for u in list(me.uv_layers): me.uv_layers.remove(u)
    uv=me.uv_layers.new(name='UVMap')
    w=W(o); li=np.zeros(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',li)
    p=w[li][:,:2]; mn=w[:,:2].min(0); mx=w[:,:2].max(0); q=(p-mn)/(mx-mn)-0.5
    S=0.010; u=(FL['x']+0.5)/H+q[:,0]*S; v=1-(FL['y']+0.5)/H+q[:,1]*S
    uv.data.foreach_set('uv',np.stack([u,v],1).ravel())
# ---- perbaikan yang sudah disepakati sebelumnya (diterapkan ulang pada geometri klaster 2)
blk=bpy.data.materials['plastik_hitam']; alu=bpy.data.materials['aluminium']
for m in bpy.data.materials:
    if m.name.startswith('plastik_hitam') and m.use_nodes:
        b=[n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'][0]
        b.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); b.inputs['Metallic'].default_value=0; b.inputs['Roughness'].default_value=0.5
        m.diffuse_color=(0.025,0.025,0.028,1)
for n in ('Rangka_Kaki_00','Rangka_Kaki_02','Rangka_Kaki_03','Rangka_Kaki_04'):
    o=bpy.data.objects[n]; o.data.materials.clear(); o.data.materials.append(blk)
for n in ('Handrest_Kanan_05','Handrest_Kiri_07','Handrest_Tengah_03'):
    o=bpy.data.objects[n]; o.data.materials.clear(); o.data.materials.append(alu)
SH=0.12
for n in ('Footrest_Kanan_Rangka','Footrest_Kiri_Rangka'):
    o=bpy.data.objects[n]; me=o.data; w=W(o)
    aluidx=[i for i,m in enumerate(me.materials) if m and m.name.startswith('aluminium')]
    pm=np.zeros(len(me.polygons),dtype=np.int32); me.polygons.foreach_get('material_index',pm)
    cnt=np.zeros(len(me.polygons),dtype=np.int32); me.polygons.foreach_get('loop_total',cnt)
    li=np.zeros(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',li)
    vm=np.full(len(w),-1,dtype=np.int32); vm[li]=np.repeat(pm,cnt)
    log('DBG',n,[m.name if m else None for m in me.materials],aluidx,np.bincount(pm).tolist(),w[:,0].min(),w[:,0].max(),int(np.isin(vm,aluidx).sum()))
    q=w[np.isin(vm,aluidx)]; q=q[(q[:,0]>w[:,0].mean()-0.2)]
    xc=np.median(q[:,0]); q=q[np.abs(q[:,0]-xc)<0.04]
    c=q.mean(0); a=np.linalg.svd(q-c,full_matrices=False)[2][0]
    if a[2]<0: a=-a
    d=w-c; tt=d@a; perp=np.linalg.norm(d-np.outer(tt,a),axis=1)
    near=(perp<0.032)&(np.abs(w[:,0]-xc)<0.045)
    tr=tt[np.isin(vm,aluidx)&near]; lo,hi=tr.min()-0.18,tr.max()+0.14
    pis=near&(tt>lo)&(tt<hi); log('piston',n,int(pis.sum()))
    w[:,2]+=SH; c2=c.copy(); c2[2]+=SH
    u=np.clip((w[:,2]-0.55)/(0.78-0.55),0,1); u=u*u*(3-2*u); wp=w[pis].copy(); w[:,2]-=SH*u
    L=c2+a*lo; T=c2+a*hi; T2=T.copy(); T2[2]-=SH
    v1=(T-L)/np.linalg.norm(T-L); v2_=(T2-L)/np.linalg.norm(T2-L)
    ax=np.cross(v1,v2_); sn=np.linalg.norm(ax); cs=float(v1@v2_); ax/=sn
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]); R=np.eye(3)+sn*K+(1-cs)*K@K
    w[pis]=(wp-L)@R.T+L
    me.vertices.foreach_set('co',w.ravel()); me.update()
def slab(name,x0,x1,y0,y1,z0,z1):
    bm=bmesh.new(); bmesh.ops.create_cube(bm,size=1.0)
    for v in bm.verts:
        v.co.x=x0 if v.co.x<0 else x1; v.co.y=y0 if v.co.y<0 else y1; v.co.z=z0 if v.co.z<0 else z1
    bmesh.ops.bevel(bm,geom=bm.edges[:],offset=0.01,segments=2,affect='EDGES')
    me=bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o=bpy.data.objects.new(name,me); bpy.context.scene.collection.objects.link(o); me.materials.append(blk)
    for p in me.polygons: p.use_smooth=True
    keep.append(o)
slab('Rangka_Dudukan_Kiri',-1.38,-0.28,-1.80,-0.76,0.571,0.60)
slab('Rangka_Dudukan_Kanan',-0.02,1.00,-1.80,-0.76,0.571,0.60)
# ---- buang semua yang bukan bagian model baru, lalu ekspor
final={o.name for o in keep}
for o in list(bpy.data.objects):
    if o.name not in final: bpy.data.objects.remove(o,do_unlink=True)
for m in [m for m in bpy.data.materials if m.users==0]: bpy.data.materials.remove(m)
log('objek akhir',len(bpy.data.objects),'face',sum(len(o.data.polygons) for o in bpy.data.objects))
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
log('DONE',OUT,os.path.getsize(OUT)/1e6)
