# Langkah D (Blender): buka hasil decimate, pasang 4 material per-label + tekstur, ekspor GLB Draco
import bpy, numpy as np, os, sys, math
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
    for _ in range(int(os.environ.get('HR_IT2','250'))):
        L=lap(U); bk=np.clip((Nrm[:,1]-0.35)/0.3,0,1)[:,None]   # hanya permukaan belakang (normal +y): sudut sandaran tidak ikut menyusut
        U=U+0.8*wu*bk*(L*Nrm).sum(1,keepdims=True)*Nrm
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
# Jaring kantong di panel kotak punggung sandaran (kedua jok): kisi belah ketupat + bingkai, mengikuti permukaan panel
if os.environ.get('NET','1')=='1':
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg=bpy.context.evaluated_depsgraph_get(); tree=BVHTree.FromObject(o,dg); MWi=o.matrix_world.inverted(); MW=o.matrix_world
    nm_=bpy.data.materials.new('jaring'); nm_.use_nodes=True; nb=nm_.node_tree.nodes['Principled BSDF']
    nb.inputs['Base Color'].default_value=(0.07,0.075,0.09,1); nb.inputs['Roughness'].default_value=0.8
    bm_=bpy.data.materials.new('jaring_dasar'); bm_.use_nodes=True; bb_=bm_.node_tree.nodes['Principled BSDF']; bb_.inputs['Base Color'].default_value=(0.01,0.011,0.015,1); bb_.inputs['Roughness'].default_value=0.9
    CELL=0.02; OFF=0.004; HW=0.0022; HD=0.0035
    def surf(x,z):
        h=tree.ray_cast(MWi@Vector((x,2.0,z)),MWi.to_3x3()@Vector((0,-1,0)))
        return None if h[0] is None else (MW@h[0]).y
    def box(V,F,p0,p1,w,d):
        t=(p1-p0); t.normalize(); n=Vector((0,1,0)); b=t.cross(n); b.normalize(); n2=b.cross(t); n2.normalize()
        base=len(V)
        for p in (p0,p1):
            for sb,sn in ((-1,-1),(1,-1),(1,1),(-1,1)): V.append(tuple(p+b*(w/2*sb)+n2*(d/2*sn)))
        for i in range(4):
            j=(i+1)%4; F+= [(base+i,base+j,base+4+j),(base+i,base+4+j,base+4+i)]
        F+= [(base,base+2,base+1),(base,base+3,base+2),(base+4,base+5,base+6),(base+4,base+6,base+7)]
    for k,(xa,xb) in enumerate(((0.17,0.65),(-0.65,-0.17))):
        za,zb=-0.285,0.015; nx=int(round((xb-xa)/CELL)); nz=int(round((zb-za)/CELL)); dx=(xb-xa)/nx; dz=(zb-za)/nz
        P={}
        for i in range(nx+1):
            for j in range(nz+1):
                x=xa+i*dx; z=za+j*dz; y=surf(x,z)
                if y is not None: P[(i,j)]=Vector((x,y+OFF,z))
        V=[];F=[]
        for (i,j),p in P.items():
            if (i+j)%2: continue
            for di,dj in ((1,1),(1,-1)):
                q=P.get((i+di,j+dj))
                if q is not None and 0<=j+dj<=nz: box(V,F,p,q,HW*2,HD*2)
        for j in (0,nz):    # bingkai atas/bawah
            for i in range(nx):
                if (i,j) in P and (i+1,j) in P: box(V,F,P[(i,j)],P[(i+1,j)],0.010,HD*2.4)
        for i in (0,nx):    # bingkai kiri/kanan
            for j in range(nz):
                if (i,j) in P and (i,j+1) in P: box(V,F,P[(i,j)],P[(i,j+1)],0.010,HD*2.4)
        nf=len(F); FM=[0]*nf
        for i in range(nx):    # alas gelap di belakang kisi
            for j in range(nz):
                c=[P.get((i,j)),P.get((i+1,j)),P.get((i+1,j+1)),P.get((i,j+1))]
                if None in c: continue
                b0=len(V); V.extend(tuple(q-Vector((0,OFF*0.6,0))) for q in c); F+= [(b0,b0+1,b0+2),(b0,b0+2,b0+3)]; FM+=[1,1]
        jm=bpy.data.meshes.new('JaringMesh%d'%k); jm.from_pydata(V,[],F); jm.update(); jm.materials.append(nm_); jm.materials.append(bm_)
        jm.polygons.foreach_set('material_index',FM)
        for pl in jm.polygons: pl.use_smooth=False
        jo=bpy.data.objects.new('Jaring_Kanan' if k==0 else 'Jaring_Kiri',jm); bpy.context.scene.collection.objects.link(jo)
        print('jaring',k,len(F),'tri')
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
HB=os.environ.get('HAND_BLEND')   # blend berisi handrest kiri baru (rakitan beberapa objek) pengganti Handrest_Kiri
sm=None
if TH:
    sm=bpy.data.materials.new('leather_stitch'); sm.use_nodes=True
    b=sm.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.03,0.032,0.036,1); b.inputs['Roughness'].default_value=0.55
    z=np.load(TH); V=z['V']; F=z['F']
    qpart=np.load(os.environ['QPART_NPY']) if os.environ.get('QPART_NPY') else np.zeros(len(V)//4,np.int8)   # bagian per quad (dihitung di luar Blender: src/newseat_qpart.py)
    fpart=qpart[np.arange(len(F))//2]     # 2 segitiga per quad
    for k in sorted(set(fpart.tolist())):
        if HB and k==2: continue   # jahitan handrest kiri lama diganti handrest baru
        sel=F[fpart==k]; used=np.unique(sel); remap=-np.ones(len(V),np.int64); remap[used]=np.arange(len(used))
        tm=bpy.data.meshes.new('JahitanMesh%d'%k); tm.from_pydata(V[used].tolist(),[],remap[sel].tolist()); tm.update()
        to=bpy.data.objects.new('Jahitan' if k==0 else 'Jahitan_'+NAMES[k],tm); bpy.context.scene.collection.objects.link(to)
        tm.materials.append(sm)
        for pl in tm.polygons: pl.use_smooth=True
if PN:
    import bmesh
    for k,nm in NAMES.items():
        if HB and k==2: continue
        bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=k],context='FACES')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        nm_=bpy.data.meshes.new(nm); bm.to_mesh(nm_); bm.free()
        for m in me.materials: nm_.materials.append(m)
        ob=bpy.data.objects.new(nm,nm_); bpy.context.scene.collection.objects.link(ob)
        print('bagian',nm,len(nm_.polygons))
    bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    seam=[e for e in bm.edges if len(e.link_faces)>=2 and any(part[f.index]==0 for f in e.link_faces) and any(part[f.index]!=0 for f in e.link_faces)]
    bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=0],context='FACES')
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
if HB:   # handrest kiri baru dari rakitan di blend pengguna (nama Handrest_Kiri_NN supaya animasi lepas/pasang di web tetap jalan)
    RATIO=float(os.environ.get('HAND_RATIO','0.12'))
    with bpy.data.libraries.load(HB) as (d0,d1): d1.objects=[n for n in d0.objects if n=='Assem--KAKI BANGKU SUZUKI' or n.startswith(('(OPSI','AS ROTOR','BESI AR','HOUSING','KARET','KEPALA SAFETY','PART','PUSH BOTTOM','ROTOR'))]
    for ob in d1.objects: bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    import bmesh
    kids=[ob for ob in d1.objects if ob.type=='MESH']; tot=0
    GELAP={'defaultplastic.001','defaultplastic.002','defaultplastic.004','defaultplastic.008','satinfinishaluminum','satinfinishaluminum.001','Material_5','Material_6'}   # abu-abu & silver -> hitam
    for i,ob in enumerate(kids):
        M=ob.matrix_world.copy(); ob.parent=None; ob.matrix_world=M
        bpy.context.view_layer.objects.active=ob; ob.select_set(True)
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        n0=len(ob.data.polygons)
        # rapikan: gabung bidang rata (CAD) dulu tanpa kehilangan bentuk, baru kurangi sisanya
        bm=bmesh.new(); bm.from_mesh(ob.data)
        bmesh.ops.dissolve_limit(bm,angle_limit=math.radians(float(os.environ.get('HAND_DISS','0.6'))),verts=bm.verts,edges=bm.edges,delimit={'MATERIAL'})
        bmesh.ops.triangulate(bm,faces=bm.faces); bm.to_mesh(ob.data); bm.free()
        ob.data.update(); n1=len(ob.data.polygons)
        tgt=int(os.environ.get('HAND_TARGET','70000'))*n0/sum(len(k.data.polygons) for k in kids) if False else None
        ob.name='Handrest_Kiri_%02d'%i; print('handrest baru',ob.name,n0,'->',n1)
    allf=sum(len(k.data.polygons) for k in kids); RATIO=min(1.0,float(os.environ.get('HAND_TARGET','120000'))/allf); print('setelah dissolve',allf,'rasio',round(RATIO,3))
    for ob in kids:
        bpy.context.view_layer.objects.active=ob
        if RATIO<1 and len(ob.data.polygons)>1500:
            md=ob.modifiers.new('dec','DECIMATE'); md.ratio=RATIO; md.delimit={'NORMAL','MATERIAL'}; md.angle_limit=math.radians(25); bpy.ops.object.modifier_apply(modifier='dec')
        me_=ob.data
        for pl in me_.polygons: pl.use_smooth=True
        try: me_.shade_smooth()
        except Exception: pass
        for sl,m in enumerate(me_.materials):
            if m and m.name in GELAP:
                nm=m.copy(); nm.name='hitam_'+m.name; nm.use_nodes=True; nt=nm.node_tree; nt.nodes.clear()
                out=nt.nodes.new('ShaderNodeOutputMaterial'); bs=nt.nodes.new('ShaderNodeBsdfPrincipled'); bs.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); bs.inputs['Roughness'].default_value=0.5; bs.inputs['Metallic'].default_value=0.0
                nt.links.new(bs.outputs['BSDF'],out.inputs['Surface']); me_.materials[sl]=nm
        tot+=len(me_.polygons); print('hasil',ob.name,len(me_.polygons))
    for ob in d1.objects:
        if ob.type=='EMPTY': bpy.data.objects.remove(ob)
    print('handrest baru total face',tot)
    # isi padat: rumah handrest berongga di sisi dalam (nampak kopong dari depan) -> tutup dengan lempeng padat sampai badan kursi
    if os.environ.get('HAND_ISI','1')=='1':
        import mathutils
        def span(o,ax): 
            vs=[v.co[ax] for v in o.data.vertices]; return max(vs)-min(vs), max(vs)
        cand=[o for o in kids if max(v.co.z for v in o.data.vertices)<-0.25]
        hs=max(cand,key=lambda o:span(o,1)[0]) if cand else None
        if hs:
            mx=max(v.co.x for v in hs.data.vertices); XB=float(os.environ.get('ISI_XB','-0.66')); TH=float(os.environ.get('ISI_TH','0.05'))
            pts=[v.co.copy() for v in hs.data.vertices if v.co.x>mx-TH][::3]
            bm=bmesh.new()
            for q in pts: bm.verts.new(q); bm.verts.new((XB,q.y,q.z))
            bmesh.ops.convex_hull(bm,input=list(bm.verts))
            for f in list(bm.faces): f.smooth=True
            hm=bpy.data.meshes.new('IsiRumahHandrest'); bm.to_mesh(hm); bm.free()
            ho=bpy.data.objects.new('Handrest_Kiri_Isi',hm); bpy.context.scene.collection.objects.link(ho)
            for pl in hm.polygons: pl.use_smooth=True
            mt=bpy.data.materials.new('isi_hitam'); mt.use_nodes=True; bs=mt.node_tree.nodes.get('Principled BSDF')
            bs.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); bs.inputs['Roughness'].default_value=0.5
            hm.materials.append(mt); print('isi rumah handrest',hs.name,len(hm.polygons),'x',round(mx,3),'->',XB)
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
