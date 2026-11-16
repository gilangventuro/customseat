# Langkah D (Blender): buka hasil decimate, pasang 4 material per-label + tekstur, ekspor GLB Draco
import bpy, numpy as np, os, sys
a=sys.argv[sys.argv.index('--')+1:]
BLEND,LAB,BASE,NORM,OUT=a
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
lab=np.load(LAB).astype(np.int32)
# Tuas merah di celah kursi: duplikat persis (geser x +DX, tanpa cermin) ke kursi sebelahnya bila LEVER_NPY diberikan
RED=os.environ.get('LEVER_NPY')
if RED:
    import bmesh
    DX=float(os.environ.get('LEVER_DX','0.827')); sel=np.load(RED); me0=o.data
    bm=bmesh.new(); bm.from_mesh(me0); bm.faces.ensure_lookup_table()
    uvl=bm.loops.layers.uv.active
    vmap={}
    def mv(v):
        if v.index not in vmap:
            vmap[v.index]=bm.verts.new((v.co.x+DX,v.co.y,v.co.z))
        return vmap[v.index]
    src=[bm.faces[i] for i in sel]
    newf=[]
    for f in src:
        vs=[mv(v) for v in f.verts]
        nf=bm.faces.new(vs)
        for l_new,l_old in zip(nf.loops, list(f.loops)):
            l_new[uvl].uv=l_old[uvl].uv
        newf.append(nf)
    bm.to_mesh(me0); bm.free(); me0.update()
    print('lever faces mirrored',len(sel))
    lab=np.concatenate([lab,lab[sel]])
me=o.data
# Haluskan permukaan atas sandaran (kerutan/duri hasil Meshy). Taubin (lambda|mu) supaya tidak menyusut; hanya z>ZLO (jahitan semuanya di bawah z=0.6)
if os.environ.get('HR_SMOOTH','1')=='1':
    ZLO=float(os.environ.get('HR_ZLO','0.60')); ZHI=float(os.environ.get('HR_ZHI','0.67')); IT=int(os.environ.get('HR_IT','80'))
    nv=len(me.vertices); P=np.empty(nv*3,np.float64); me.vertices.foreach_get('co',P); P=P.reshape(-1,3)
    ed=np.empty(len(me.edges)*2,np.int32); me.edges.foreach_get('vertices',ed); ed=ed.reshape(-1,2)
    # titik kembar (posisi sama, terputus di jahitan/UV) disatukan dulu supaya celah retak ikut tertutup
    _,uid=np.unique(np.round(P*1e5).astype(np.int64),axis=0,return_inverse=True); uid=uid.ravel(); nu=uid.max()+1
    print('titik',nv,'unik',nu)
    a_=np.concatenate([uid[ed[:,0]],uid[ed[:,1]]]); b_=np.concatenate([uid[ed[:,1]],uid[ed[:,0]]])
    k_=np.unique(a_.astype(np.int64)*nu+b_); a_=(k_//nu).astype(np.int64); b_=(k_%nu).astype(np.int64)
    deg=np.bincount(a_,minlength=nu).astype(np.float64)
    w=np.clip((P[:,2]-ZLO)/(ZHI-ZLO),0,1)[:,None]; w=w*w*(3-2*w)
    def lap(Q):
        s=np.zeros((nu,3))
        for c in range(3): s[:,c]=np.bincount(a_,weights=Q[b_,c],minlength=nu)
        return np.where(deg[:,None]>0,s/np.maximum(deg,1)[:,None]-Q,0)
    MU=float(os.environ.get('HR_MU','0.53')); cnt=np.bincount(uid,minlength=nu).astype(np.float64)
    U=np.zeros((nu,3))
    for c in range(3): U[:,c]=np.bincount(uid,weights=P[:,c],minlength=nu)/cnt
    wu=np.zeros((nu,1)); wu[uid]=w
    for _ in range(IT):
        U=U+0.5*wu*lap(U); U=U-MU*wu*lap(U)
    # tahap 2: haluskan hanya searah normal (menutup alur sempit/kerutan tipis yang tidak hilang oleh Laplacian biasa)
    Nrm=np.zeros((nu,3)); me.update(); vn0=np.empty(nv*3); me.vertex_normals.foreach_get('vector',vn0); vn0=vn0.reshape(-1,3)
    for c in range(3): Nrm[:,c]=np.bincount(uid,weights=vn0[:,c],minlength=nu)/cnt
    for _ in range(20): Nrm=Nrm+0.5*lap(Nrm)
    Nrm/=np.maximum(np.linalg.norm(Nrm,axis=1,keepdims=True),1e-9)
    for _ in range(int(os.environ.get('HR_IT2','0'))):
        L=lap(U); U=U+0.8*wu*(L*Nrm).sum(1,keepdims=True)*Nrm
    P=P+(U[uid]-np.array([np.bincount(uid,weights=P[:,c],minlength=nu)/cnt for c in range(3)]).T[uid])*(w>0)
    me.vertices.foreach_set('co',P.ravel()); me.update()
    # normal bawaan (custom) masih memuat kerutan lama -> ganti dengan normal geometri baru di area yang dihaluskan
    if me.has_custom_normals:
        nl=len(me.loops); li=np.empty(nl,np.int32); me.loops.foreach_get('vertex_index',li)
        old_n=np.empty(nl*3,np.float64); me.corner_normals.foreach_get('vector',old_n); old_n=old_n.reshape(-1,3)
        vn=np.empty(nv*3,np.float64); me.vertex_normals.foreach_get('vector',vn); vn=vn.reshape(-1,3)
        wl=w[li]; nn=old_n*(1-wl)+vn[li]*wl; nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-9)
        me.normals_split_custom_set(nn.tolist()); me.update()
    print('haluskan sandaran: verts',int((w[:,0]>0).sum()),'iter',IT)
me.materials.clear()
base=bpy.data.images.load(BASE); norm=bpy.data.images.load(NORM); norm.colorspace_settings.name='Non-Color'
def mk(name,rough):
    m=bpy.data.materials.new(name); m.use_nodes=True
    nt=m.node_tree; nt.nodes.clear()
    out=nt.nodes.new('ShaderNodeOutputMaterial'); bs=nt.nodes.new('ShaderNodeBsdfPrincipled')
    bs.inputs['Roughness'].default_value=rough; bs.inputs['Metallic'].default_value=0.0
    tb=nt.nodes.new('ShaderNodeTexImage'); tb.image=base
    tn=nt.nodes.new('ShaderNodeTexImage'); tn.image=norm
    nm=nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value=1.0
    nt.links.new(tb.outputs['Color'],bs.inputs['Base Color'])
    nt.links.new(tn.outputs['Color'],nm.inputs['Color']); nt.links.new(nm.outputs['Normal'],bs.inputs['Normal'])
    nt.links.new(bs.outputs['BSDF'],out.inputs['Surface'])
    return m
for name,r in [('leather_main',0.55),('leather_pattern',0.55),('leather_accent',0.55),('leather_rest',0.6)]:
    me.materials.append(mk(name,r))
lab=np.where(lab==4,0,lab)   # jahitan = geometri benang terpisah (THREAD_NPZ), bukan label material
me.polygons.foreach_set('material_index',lab); me.update()
PN=os.environ.get('PART_NPY'); NAMES={1:'Handrest_Kanan',2:'Handrest_Kiri',3:'Footrest_Kanan',4:'Footrest_Kiri',5:'Handrest_Tengah'}
part=np.zeros(len(me.polygons),np.int8)
if PN:
    pp=np.load(PN); m_=min(len(pp),len(part)); part[:m_]=pp[:m_]
TH=os.environ.get('THREAD_NPZ')
sm=None
if TH:
    sm=bpy.data.materials.new('leather_stitch'); sm.use_nodes=True
    b=sm.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.03,0.032,0.036,1); b.inputs['Roughness'].default_value=0.55
    z=np.load(TH); V=z['V']; F=z['F']
    qpart=np.load(os.environ['QPART_NPY']) if os.environ.get('QPART_NPY') else np.zeros(len(V)//4,np.int8)   # bagian per quad (dihitung di luar Blender: src/newseat_qpart.py)
    fpart=qpart[np.arange(len(F))//2]     # 2 segitiga per quad
    for k in sorted(set(fpart.tolist())):
        sel=F[fpart==k]; used=np.unique(sel); remap=-np.ones(len(V),np.int64); remap[used]=np.arange(len(used))
        tm=bpy.data.meshes.new('JahitanMesh%d'%k); tm.from_pydata(V[used].tolist(),[],remap[sel].tolist()); tm.update()
        to=bpy.data.objects.new('Jahitan' if k==0 else 'Jahitan_'+NAMES[k],tm); bpy.context.scene.collection.objects.link(to)
        tm.materials.append(sm)
        for pl in tm.polygons: pl.use_smooth=True
if PN:
    import bmesh
    for k,nm in NAMES.items():
        bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=k],context='FACES')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        nm_=bpy.data.meshes.new(nm); bm.to_mesh(nm_); bm.free()
        for m in me.materials: nm_.materials.append(m)
        ob=bpy.data.objects.new(nm,nm_); bpy.context.scene.collection.objects.link(ob)
        print('bagian',nm,len(nm_.polygons))
    bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    seam=[e for e in bm.edges if len(e.link_faces)>=2 and any(part[f.index]==0 for f in e.link_faces) and any(part[f.index]!=0 for f in e.link_faces)]
    FB=os.environ.get('FRAME_BLEND'); ZC=float(os.environ.get('FRAME_ZC','-0.62'))
    dele=lambda f: part[f.index]!=0 or (FB and all(v.co.z<ZC for v in f.verts))   # rangka lama (di bawah ZC) dibuang bila rangka baru dipasang
    bmesh.ops.delete(bm,geom=[f for f in bm.faces if dele(f)],context='FACES')
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
    # tutup lubang bekas bagian yang dilepas (bagian dalam badan jadi tidak terlihat bolong)
    seam=[e for e in seam if e.is_valid and e.is_boundary]
    capf=bmesh.ops.holes_fill(bm,edges=seam,sides=4000)['faces']
    seam2=[e for e in seam if e.is_valid and e.is_boundary]
    try: capf+= [g for g in bmesh.ops.triangle_fill(bm,edges=seam2,use_beauty=True)['geom'] if isinstance(g,bmesh.types.BMFace)]
    except Exception as ex: print('triangle_fill gagal',ex)
    print('sisa tepi terbuka',len([e for e in seam2 if e.is_valid and e.is_boundary]))
    for f in capf: f.material_index=3; f.smooth=True
    print('tutup lubang: sisi',len(seam),'face penutup',len(capf))
    bm.to_mesh(me); bm.free()
FB=os.environ.get('FRAME_BLEND')
if FB:   # rangka baru dari scan (objek 'Rangka'), material grey
    with bpy.data.libraries.load(FB) as (d0,d1): d1.objects=['Rangka']
    ro=d1.objects[0]; bpy.context.scene.collection.objects.link(ro); ro.name='Rangka'
    rm_=bpy.data.materials.new('rangka_grey'); rm_.use_nodes=True; bb=rm_.node_tree.nodes['Principled BSDF']
    bb.inputs['Base Color'].default_value=(0.30,0.31,0.33,1); bb.inputs['Roughness'].default_value=0.5; bb.inputs['Metallic'].default_value=0.4
    ro.data.materials.clear(); ro.data.materials.append(rm_); print('rangka baru',len(ro.data.polygons))
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
