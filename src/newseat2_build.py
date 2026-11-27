# Bangun GLB situs dari "SEATS ALLDILA UPDATE 2.blend" klaster 2 (seat Meshy baru: Mesh_0 / Mesh_0.001 + semua aksesori sudah dipasang user).
# Jalankan: blender --background --python src/newseat2_build.py -- <out.glb>   (butuh _tmp_tex/* dari newseat2_classmap.py & newseat2_tex.py)
import bpy,bmesh,numpy as np,json,math,sys,re,os
P='C:/Users/LENOVO LOQ/Projects/alldila-jok-simulator/'
OUT=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else P+'_tmp_new2.glb'
DX=0.0; DEC=float(os.environ.get('DEC','0.5'))
def log(*a): print(*a,flush=True)
bpy.ops.wm.open_mainfile(filepath=P+'SEATS ALLDILA UPDATE 2.blend')
for m in [m for m in bpy.data.materials if m.users==0]: bpy.data.materials.remove(m)
names={o.name for o in bpy.data.objects if o.type=='MESH'}
for o in bpy.data.objects:
    if o.type=='MESH':
        bpy.context.view_layer.objects.active=o
        if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
def strip(n): return re.sub(r'\.\d{3}$','',n)
canon={}
def _mx(o):
    M=np.array(o.matrix_world); v=np.zeros(len(o.data.vertices)*3); o.data.vertices.foreach_get('co',v); return float((v.reshape(-1,3)@M[:3,:3].T+M[:3,3])[:,0].mean())
for n in names:
    if n.startswith('Kubus') or n=='Handrest_Tengah_05.002' or n.startswith('Jaring'): continue
    if abs(_mx(bpy.data.objects[n]))>1.6: continue   # klaster lain di file (x jauh di kiri) tidak dipakai
    canon[n]=strip(n)
for b in ('Sandaran_Belakang_Kiri_02','Handle_Kiri_00','Cupholder_Kiri_00'):
    for suf,side in (('.003','Kiri'),('.004','Kanan')):
        if b+suf in canon: canon[b+suf]=b.replace('Kiri',side)
canon['Mesh_0.001']='Kursi_Kiri'; canon['Mesh_0.003']='Kursi_Kanan'
log('objek model',len(canon))
# bahan bersufiks (.003/.004...) digabung ke satu bahan kanonik; dua kayu berbeda tetap dua
WOODS={}
def canon_mat(m):
    b=strip(m.name)
    if b=='satinfinishwalnut':
        WOODS.setdefault(m.name,len(WOODS)); return 'satinfinishwalnut' if WOODS[m.name]==0 else 'satinfinishwalnut.001'
    return b
cm={}
for m in list(bpy.data.materials):
    cm.setdefault(canon_mat(m),[]).append(m)
canonmat={}
for k,vs in cm.items():
    main=vs[0]
    for o in vs[1:]: o.user_remap(main)
    canonmat[k]=main
for k,main in canonmat.items():
    for o in list(bpy.data.materials):
        if o is not main and o.name==k: o.name=k+'_x'
    main.name=k
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
    if o.name not in _ks: bpy.data.objects.remove(o,do_unlink=True)
# ---- bahan kulit bertekstur baru
for nm in ('leather_main','leather_pattern','leather_accent','leather_stitch'):
    if nm in bpy.data.materials: bpy.data.materials[nm].name='old_'+nm
tint=bpy.data.images.load(P+'_tmp_tex/base2048.jpg'); tint.colorspace_settings.name='sRGB'   # albedo asli (warna bawaan = seperti di Blender); tint-base abu-abu dimuat terpisah oleh situs
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
# logam (tutup engsel armrest ber-logo, dll.): situs tanpa peta lingkungan -> logam penuh tampak hitam; turunkan metalik agar terlihat abu perak seperti di Blender
for _n,_mv in (('baja',0.25),('aluminium',0.25)):
    _m=canonmat.get(_n)
    if _m is not None and _m.use_nodes:
        for _b in [x for x in _m.node_tree.nodes if x.type=='BSDF_PRINCIPLED']: _b.inputs['Metallic'].default_value=_mv
CL=np.load(P+'_tmp_tex/classmap.npy'); H=CL.shape[0]; FL=json.load(open(P+'_tmp_tex/flatuv2.json'))
# ---- badan kursi baru: decimate + kelas per-face dari peta kelas tekstur
from mathutils import kdtree
for k in ('Mesh_0.001','Mesh_0.003'):
    o=bpy.data.objects[k]; nn=canon[k]; o.name=nn; o.data.name=nn
    bpy.context.view_layer.objects.active=o
    n0=len(o.data.polygons)
    # kelas per-face = 3 material yang dipisah user di Blender (Master kursi / Bahan / List -> main / pattern / accent)
    mi0=np.zeros(n0,dtype=np.int32); o.data.polygons.foreach_get('material_index',mi0)
    c0=np.zeros(n0*3); o.data.polygons.foreach_get('center',c0); c0=c0.reshape(-1,3)
    md=o.modifiers.new('dec','DECIMATE'); md.ratio=DEC; md.delimit={'UV'}
    bpy.ops.object.modifier_apply(modifier='dec')
    me=o.data
    while len(me.color_attributes): me.color_attributes.remove(me.color_attributes[0])
    n1=len(me.polygons); c1=np.zeros(n1*3); me.polygons.foreach_get('center',c1); c1=c1.reshape(-1,3)
    kd=kdtree.KDTree(n0)
    for i in range(n0): kd.insert(c0[i],i)
    kd.balance()
    fc=np.array([mi0[kd.find(c1[i])[1]] for i in range(n1)],dtype=np.int32)
    # buang pulau kecil (face nyasar berbeda material di tengah panel): komponen < MINF face dilebur ke kelas mayoritas yang mengelilinginya
    _bm=bmesh.new(); _bm.from_mesh(me); _bm.faces.ensure_lookup_table()
    adj=[[f2.index for e in f.edges for f2 in e.link_faces if f2 is not f] for f in _bm.faces]; _bm.free()
    MINF=int(os.environ.get('MINF','400'))
    for _ in range(3):
        seen=np.zeros(n1,bool); changed=0
        for st0 in range(n1):
            if seen[st0]: continue
            cl=fc[st0]; comp=[st0]; seen[st0]=True; q=[st0]
            while q:
                u=q.pop()
                for w2 in adj[u]:
                    if not seen[w2] and fc[w2]==cl: seen[w2]=True; comp.append(w2); q.append(w2)
            if len(comp)<MINF:
                cs=set(comp); bc=np.zeros(3,int)
                for u in comp:
                    for w2 in adj[u]:
                        if w2 not in cs: bc[fc[w2]]+=1
                if bc.sum(): 
                    for u in comp: fc[u]=bc.argmax()
                    changed+=1
        log('pulau kecil dilebur',changed)
        if not changed: break
    me.materials.clear()
    for nm in ('leather_main','leather_pattern','leather_accent'): me.materials.append(LM[nm])
    me.polygons.foreach_set('material_index',fc)
    for p in me.polygons: p.use_smooth=True
    log('seat',nn,n0,'->',len(me.polygons),np.bincount(fc,minlength=3).tolist())
# ---- nama objek kanonik + bahan kulit baru untuk bantalan legrest
for o in keep:
    if o.name.startswith('Kursi_'): continue
    nn=canon[o.name]
    for i,sl in enumerate(o.material_slots):
        t=sl.material
        if t is not None and t.name in LM: o.data.materials[i]=LM[t.name]
        elif t is not None and t.name.startswith('old_'): o.data.materials[i]=LM[t.name[4:]]
    o.name=nn; o.data.name=nn
# ---- jaring yang oleh user sudah digabung ke cangkang -> pisah jadi objek sendiri (Jaring_<sisi>_Bingkai/Karet/Net) supaya tetap bisa dilepas
NETN={'net_frame':'Bingkai','net_band':'Karet','net_mesh':'Net'}
for side in ('Kiri','Kanan'):
    sh=bpy.data.objects['Sandaran_Belakang_%s_02'%side]
    for idx,sl in enumerate(list(sh.material_slots)):
        base=sl.material.name if sl.material else ''
        if base not in NETN: continue
        d=sh.copy(); d.data=sh.data.copy(); bpy.context.scene.collection.objects.link(d)
        bm=bmesh.new(); bm.from_mesh(d.data)
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.material_index!=idx],context='FACES')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        bm.to_mesh(d.data); bm.free()
        d.name='Jaring_%s_%s'%(side,NETN[base]); d.data.name=d.name
        me=d.data; me.materials.clear(); me.materials.append(sl.material)
        if NETN[base]=='Net':   # UV planar dalam meter (tekstur jaring di situs berulang tiap 0,028 m)
            for u in list(me.uv_layers): me.uv_layers.remove(u)
            uv=me.uv_layers.new(name='UVMap'); li=np.zeros(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',li)
            w=W(d)[li]; uv.data.foreach_set('uv',np.stack([w[:,0],w[:,2]],1).ravel())
        keep.append(d)
    bm=bmesh.new(); bm.from_mesh(sh.data)
    bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.material_index!=0],context='FACES')
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
    bm.to_mesh(sh.data); bm.free()
    sh.data.materials.clear(); sh.data.materials.append(canonmat['plastik_hitam'])
for nn in ('Footrest_Kiri_Bantalan','Footrest_Kanan_Bantalan'):
    o=bpy.data.objects[nn]; me=o.data
    for u in list(me.uv_layers): me.uv_layers.remove(u)
    uv=me.uv_layers.new(name='UVMap')
    w=W(o); li=np.zeros(len(me.loops),dtype=np.int32); me.loops.foreach_get('vertex_index',li)
    cnt=np.zeros(len(me.polygons),dtype=np.int32); me.polygons.foreach_get('loop_total',cnt)
    pm=np.zeros(len(me.polygons),dtype=np.int32); me.polygons.foreach_get('material_index',pm)
    lc=np.repeat(pm,cnt)   # kelas per loop (0 utama, 1 motif, 2 list) -> tambalan datar warna asli kelas itu
    names_=[m.name if m else '' for m in me.materials]
    cls=np.array([{'leather_main':0,'leather_pattern':1,'leather_accent':2}.get(names_[i],0) for i in range(len(names_))])[lc]
    p=w[li][:,:2]; mn=w[:,:2].min(0); mx=w[:,:2].max(0); q=(p-mn)/(mx-mn)-0.5
    u=np.zeros(len(li)); v=np.zeros(len(li))
    for c in range(3):
        pc=FL['patches'][str(c)]; S=0.7*pc['w']/H; m=cls==c
        u[m]=(pc['x']+0.5)/H+q[m,0]*S; v[m]=1-(pc['y']+0.5)/H+q[m,1]*S
    uv.data.foreach_set('uv',np.stack([u,v],1).ravel())
# ---- perbaikan yang sudah disepakati sebelumnya (diterapkan ulang pada geometri klaster 2)
blk=canonmat['plastik_hitam']; alu=canonmat['aluminium']
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
slab('Rangka_Dudukan_Kiri',-1.145,-0.045,-1.80,-0.76,0.571,0.60)
slab('Rangka_Dudukan_Kanan',0.215,1.235,-1.80,-0.76,0.571,0.60)
# alas hitam tepat di belakang jaring (seukuran jaring): menutupi dudukan bulat di sudut-sudutnya; cangkang tetap seperti file Blender
for side in ('Kiri','Kanan'):
    nt=bpy.data.objects['Jaring_%s_Net'%side]
    d=nt.copy(); d.data=nt.data.copy(); bpy.context.scene.collection.objects.link(d); d.name='Jaring_%s_Alas'%side; d.data.name=d.name
    d.data.materials.clear(); d.data.materials.append(blk)
    for u in list(d.data.uv_layers): d.data.uv_layers.remove(u)
    wd=W(d); wd[:,1]+=0.0; wd[:,1]-=float(os.environ.get('ALAS_DY','0.006')); setW(d,wd); keep.append(d)
# ---- buang semua yang bukan bagian model baru, lalu ekspor
final={o.name for o in keep}
for o in list(bpy.data.objects):
    if o.name not in final: bpy.data.objects.remove(o,do_unlink=True)
for m in [m for m in bpy.data.materials if m.users==0]: bpy.data.materials.remove(m)
log('objek akhir',len(bpy.data.objects),'face',sum(len(o.data.polygons) for o in bpy.data.objects))
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
log('DONE',OUT,os.path.getsize(OUT)/1e6)
