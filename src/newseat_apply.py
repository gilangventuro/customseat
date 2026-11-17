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
    nb.inputs['Base Color'].default_value=(0.008,0.008,0.009,1); nb.inputs['Roughness'].default_value=0.8
    bm_=bpy.data.materials.new('jaring_dasar'); bm_.use_nodes=True; bb_=bm_.node_tree.nodes['Principled BSDF']; bb_.inputs['Base Color'].default_value=(0.0,0.0,0.0,1); bb_.inputs['Roughness'].default_value=0.9
    CELL=0.02; OFF=0.003; HW=0.0022; HD=0.0035
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
    for k,(xa,xb) in enumerate(((0.155,0.665),(-0.665,-0.155))):
        za,zb=-0.30,0.015; nx=int(round((xb-xa)/CELL)); nz=int(round((zb-za)/CELL)); dx=(xb-xa)/nx; dz=(zb-za)/nz
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
# Ratakan cekungan kecil (lubang) di sisi luar atas sandaran: isi dengan permukaan halus dari tepi sekitarnya (membran Laplace)
if os.environ.get('DENT_FIX','1')=='1':
    _M=np.array(o.matrix_world); _Mi=np.linalg.inv(_M)
    _nv=len(me.vertices); _P=np.empty(_nv*3,np.float64); me.vertices.foreach_get('co',_P); _P=_P.reshape(-1,3); _W=_P@_M[:3,:3].T+_M[:3,3]
    _ed=np.empty(len(me.edges)*2,np.int32); me.edges.foreach_get('vertices',_ed); _ed=_ed.reshape(-1,2)
    _,_uid=np.unique(np.round(_W*1e5).astype(np.int64),axis=0,return_inverse=True); _uid=_uid.ravel(); _nu=_uid.max()+1
    _U=np.zeros((_nu,3)); _U[_uid]=_W
    _ea=np.concatenate([_uid[_ed[:,0]],_uid[_ed[:,1]]]); _eb=np.concatenate([_uid[_ed[:,1]],_uid[_ed[:,0]]])
    _deg=np.bincount(_ea,minlength=_nu).astype(np.float64)
    RX=float(os.environ.get('DENT_RX','0.026')); RZ=float(os.environ.get('DENT_RZ','0.034'))
    for _cx in (0.787,-0.788):
        _cz=0.358
        _r=np.sqrt(((_U[:,0]-_cx)/RX)**2+((_U[:,2]-_cz)/RZ)**2)
        _ring=np.nonzero((_r>1.5)&(_r<2.4))[0]
        if len(_ring)<5: print('dent: ring kosong',_cx); continue
        _yr=_U[_ring,1]; _yf=np.median(_yr[_yr<_yr.min()+0.04])
        _reg=np.nonzero((_r<1.4)&(_U[:,1]<_yf+0.30)&(_U[:,1]>_yf-0.03))[0]
        _m=np.zeros(_nu,bool); _m[_reg]=True
        _all=np.nonzero(_r<1.4)[0]; print('dent y semua',np.round(np.percentile(_U[_all,1],[0,5,25,50,75,95,100]),3),'yf',round(_yf,3))
        _y0=_U[_reg,1].copy()
        for _it in range(300):
            _acc=np.zeros((_nu,3)); np.add.at(_acc,_ea,_U[_eb])
            _new=_acc/np.maximum(_deg,1)[:,None]
            _U[_m]=_new[_m]
        # permukaan akhir = polinomial kubik dari cincin sekitar (tanpa bintik), memudar ke hasil membran di tepi
        _rg=_ring[_U[_ring,1]<_yf+0.1] if _cx>0 else _ring[:0]
        _uu=(_U[:,0]-_cx)/RX; _vv=(_U[:,2]-_cz)/RZ
        def _bs(u,v): return np.stack([np.ones_like(u),u,v,u*u,u*v,v*v,u**3,u*u*v,u*v*v,v**3],1)
        if len(_rg)>=10: _co,*_=np.linalg.lstsq(_bs(_uu[_rg],_vv[_rg]),_U[_rg,1],rcond=None)
        if len(_rg)>=10:
            _in=np.nonzero(_r<1.5)[0]; _yfit=_bs(_uu[_in],_vv[_in])@_co
            _wf=np.clip((1.5-_r[_in])/0.3,0,1); _okm=(_U[_in,1]<_yf+0.30)&(_U[_in,1]>_yf-0.03)
            _U[_in,1]=np.where(_okm,_U[_in,1]+_wf*(_yfit-_U[_in,1]),_U[_in,1])
        # haluskan juga pinggiran sekitarnya dengan bobot memudar supaya tidak ada kerutan/bekas
        _bd=np.nonzero((_r<3.2)&(np.abs(_U[:,1]-_yf)<0.03))[0]
        _w=np.zeros(_nu); _w[_bd]=np.clip((3.2-_r[_bd])/(3.2-1.25),0,1)
        for _it in range(int(os.environ.get('DENT_IT2','250'))):
            _acc=np.zeros((_nu,3)); np.add.at(_acc,_ea,_U[_eb]); _nw=_acc/np.maximum(_deg,1)[:,None]
            _U+=(0.5*_w)[:,None]*(_nw-_U)
        print('dent',_cx,'titik',len(_reg),'y permukaan',round(_yf,4),'geser maks',round(float(np.abs(_U[_reg,1]-_y0).max()),4))
    _W2=_U[_uid]; _P2=(_W2-_M[:3,3])@_Mi[:3,:3].T
    me.vertices.foreach_set('co',_P2.ravel()); me.update()
# Tambal lubang/kawah kecil di sisi dalam kedua jok (dekat handrest tengah): lempeng cakram halus mengikuti permukaan sekitar, UV & material dari tetangga
if os.environ.get('HOLE_FIX','1')=='1':
    import bmesh, mathutils
    _M=np.array(o.matrix_world); _Mi=np.linalg.inv(_M)
    _nv=len(me.vertices); _P=np.empty(_nv*3,np.float64); me.vertices.foreach_get('co',_P); _P=_P.reshape(-1,3); _W=_P@_M[:3,:3].T+_M[:3,3]
    bm=bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table(); uvl=bm.loops.layers.uv.active
    nf0=len(bm.faces); addlab=[]
    for _oo in (1,-1):
        for _cy,_cz,_R in [(0.246,-0.393,0.03),(0.05,-0.448,0.02),(0.279,-0.48,0.02)]:
            bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
            _side=(_W[:,0]*_oo<=0.002)&(np.abs(_W[:,0])<0.3)
            _r=np.hypot(_W[:,1]-_cy,_W[:,2]-_cz)/_R
            _ring=np.nonzero(_side&(_r>1.5)&(_r<2.8))[0]
            if len(_ring)<8: print('tambal: ring kosong',_oo,_cy,_cz); continue
            _xr=_W[_ring,0]*_oo; _rg=_ring[_xr>_xr.max()-0.015]
            if len(_rg)<8: print('tambal: ring sempit',_oo,_cy,_cz); continue
            _rg=np.array([i for i in _rg if len(bm.verts[int(i)].link_loops)],dtype=_rg.dtype)
            if len(_rg)<8: print('tambal: ring sempit(loop)',_oo,_cy,_cz); continue
            _u=(_W[_rg,1]-_cy)/_R; _v=(_W[_rg,2]-_cz)/_R
            _bs=lambda u,v: np.stack([np.ones_like(u),u,v,u*u,u*v,v*v],1)
            _co,*_=np.linalg.lstsq(_bs(_u,_v),_W[_rg,0],rcond=None)
            _uv=np.array([list(bm.verts[int(i)].link_loops[0][uvl].uv) for i in _rg])
            _A=np.stack([np.ones_like(_u),_u,_v],1); _cu,*_=np.linalg.lstsq(_A,_uv,rcond=None)
            _res=np.linalg.norm(_A@_cu-_uv,axis=1); _ok=_res<=max(2*_res.std(),1e-4)
            if _ok.sum()>=6: _cu,*_=np.linalg.lstsq(_A[_ok],_uv[_ok],rcond=None)
            labs=[int(lab[l.face.index]) for i in _rg for l in bm.verts[int(i)].link_loops if l.face.index<nf0]
            _ml=max(set(labs),key=labs.count) if labs else 0
            g=bmesh.new()
            c=bmesh.ops.create_circle(g,cap_ends=True,cap_tris=True,segments=20,radius=1.0)
            bmesh.ops.subdivide_edges(g,edges=list(g.edges),cuts=2,use_grid_fill=True)
            if _oo<0: bmesh.ops.reverse_faces(g,faces=list(g.faces))
            RP=1.35; mp={}
            for gv in g.verts:
                pu=gv.co.x*RP; pv=gv.co.y*RP
                xx=float(_bs(np.array([pu]),np.array([pv]))@_co)+0.0006*_oo
                mp[gv.index]=bm.verts.new((xx,_cy+pu*_R,_cz+pv*_R))
            g.verts.ensure_lookup_table()
            for gf in g.faces:
                nfc=bm.faces.new([mp[v.index] for v in gf.verts]); nfc.smooth=True
                for l in nfc.loops:
                    pu=(l.vert.co.y-_cy)/_R; pv=(l.vert.co.z-_cz)/_R
                    l[uvl].uv=tuple(np.array([1,pu,pv])@_cu)
                addlab.append(_ml)
            g.free()
            print('tambal',_oo,_cy,_cz,'face',len(g.faces) if False else 'ok','material',_ml)
    bm.to_mesh(me); bm.free(); me.update()
    if addlab: lab=np.concatenate([lab,np.array(addlab,dtype=lab.dtype)])
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
DN=os.environ.get('DROP_NPY')   # wajah badan yang sudah dibuang pengguna di blend baru (rangka bawah dsb.) ikut dibuang
if DN:
    _dm=np.load(DN).astype(bool)[:len(part)]; _sel=np.zeros(len(part),bool); _sel[:len(_dm)]=_dm
    _sel&=(part==0); part[_sel]=9; print('buang wajah sesuai blend baru',int(_sel.sum()))
TH=os.environ.get('THREAD_NPZ')
HB=os.environ.get('HAND_BLEND')   # blend berisi handrest kiri baru (rakitan beberapa objek) pengganti Handrest_Kiri
REAR_IN=bool(HB) and os.environ.get('REAR_INPLACE','0')=='1'   # cangkang kedua jok sudah ada di blend pengguna: pakai apa adanya
REAR=HB and os.environ.get('REAR_NEW','0')=='1' and not REAR_IN
if REAR and PN:   # area merah (label 3 = 4_TidakBisaDiganti) di belakang sandaran dibuang, diganti cangkang belakang baru
    _M=np.array(o.matrix_world); _nv=len(me.vertices); _P=np.empty(_nv*3); me.vertices.foreach_get('co',_P); _P=_P.reshape(-1,3); _W=_P@_M[:3,:3].T+_M[:3,3]
    _li=np.empty(len(me.loops),np.int32); me.loops.foreach_get('vertex_index',_li); _ls=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('loop_start',_ls); _lt=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('loop_total',_lt)
    _c=np.zeros((len(me.polygons),3))
    for _k in range(int(_lt.max())):
        _m=_lt>_k; _c[_m]+=_W[_li[_ls[_m]+_k]]
    _c/=_lt[:,None]
    DXR=float(os.environ.get('LEVER_DX','0.827')); X0=float(os.environ.get('REAR_X0','-0.84')); X1=float(os.environ.get('REAR_X1','0.03')); Z0=float(os.environ.get('REAR_Z0','-0.37')); Z1=float(os.environ.get('REAR_Z1','0.62')); Y0=float(os.environ.get('REAR_Y0','0.15'))
    MXR=float(os.environ.get('REAR_MX','0.0'))   # sisakan pinggir (dinding samping) supaya tidak bolong di samping cangkang
    _rm=np.zeros(len(me.polygons),bool)
    for _dx in (0.0,DXR):
        _rm|=(lab[:len(_c)]==3)&(_c[:,0]>X0+_dx+MXR)&(_c[:,0]<X1+_dx-MXR)&(_c[:,2]>Z0)&(_c[:,2]<Z1)&(_c[:,1]>Y0)&(part==0)
    if os.environ.get('REAR_EXCL_NPY'):   # wajah yang bukan bagian cangkang (rangka dll. di blend pengguna) jangan ikut dibuang
        _ex=np.zeros(len(_rm),bool); _e0=np.load(os.environ['REAR_EXCL_NPY']).astype(bool)[:len(_rm)]; _ex[:len(_e0)]=_e0
        _rm0=_rm.copy(); _rm[:]=False
        for _dx in (0.0,DXR):   # di area punggung: buang semua wajah yang sudah dibuang pengguna + area merah; di luar area (rangka) dipertahankan
            _bx=(_c[:,0]>X0+_dx+MXR)&(_c[:,0]<X1+_dx-MXR)&(_c[:,2]>Z0)&(_c[:,2]<Z1)&(_c[:,1]>Y0)&(part==0)
            _rm|=_bx&(_ex|(lab[:len(_c)]==3))
    part[_rm]=9; print('area merah belakang dibuang',int(_rm.sum()),'face')
    RC=_c[_rm&(_c[:,0]<X1-MXR)&(_c[:,0]>X0+MXR)]   # pusat face area merah jok kiri (untuk menempelkan cangkang baru)
sm=None
if TH:
    sm=bpy.data.materials.new('leather_stitch'); sm.use_nodes=True
    b=sm.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.03,0.032,0.036,1); b.inputs['Roughness'].default_value=0.55
    z=np.load(TH); V=z['V']; F=z['F']
    qpart=np.load(os.environ['QPART_NPY']) if os.environ.get('QPART_NPY') else np.zeros(len(V)//4,np.int8)   # bagian per quad (dihitung di luar Blender: src/newseat_qpart.py)
    fpart=qpart[np.arange(len(F))//2]     # 2 segitiga per quad
    for k in sorted(set(fpart.tolist())):
        if HB and k in (1,2,5): continue   # jahitan handrest kiri lama diganti handrest baru
        sel=F[fpart==k]; used=np.unique(sel); remap=-np.ones(len(V),np.int64); remap[used]=np.arange(len(used))
        tm=bpy.data.meshes.new('JahitanMesh%d'%k); tm.from_pydata(V[used].tolist(),[],remap[sel].tolist()); tm.update()
        to=bpy.data.objects.new('Jahitan' if k==0 else 'Jahitan_'+NAMES[k],tm); bpy.context.scene.collection.objects.link(to)
        tm.materials.append(sm)
        for pl in tm.polygons: pl.use_smooth=True
if PN:
    import bmesh
    for k,nm in NAMES.items():
        if HB and k in (1,2,5): continue
        bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=k],context='FACES')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        nm_=bpy.data.meshes.new(nm); bm.to_mesh(nm_); bm.free()
        for m in me.materials: nm_.materials.append(m)
        ob=bpy.data.objects.new(nm,nm_); bpy.context.scene.collection.objects.link(ob)
        print('bagian',nm,len(nm_.polygons))
    bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    seam=[e for e in bm.edges if len(e.link_faces)>=2 and any(part[f.index]==0 for f in e.link_faces) and any(part[f.index] not in (0,9) for f in e.link_faces)]
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

for SIDE,ASM,K,SGN in [("Kiri","Assem--KAKI BANGKU SUZUKI",2,-1),("Kanan","Assem2",1,1),("Tengah","Assem1--OPSI 3",5,0)]:
    if HB:   # handrest baru dari rakitan di blend pengguna (nama Handrest_Kiri_NN supaya animasi lepas/pasang di web tetap jalan)
        RATIO=float(os.environ.get('HAND_RATIO','0.12'))
        if 'ALLOBJ' not in globals():
            with bpy.data.libraries.load(HB) as (d0,d1_): d1_.objects=[n for n in d0.objects if n not in ('Kursi','Kursi MESH','Mesh_0.001')]
            ALLOBJ=[o for o in d1_.objects if o]
            SELS={a:[o for o in ALLOBJ if o.name==a or (o.parent and o.parent.name==a)] for a in ('Assem--KAKI BANGKU SUZUKI','Assem2','Assem1--OPSI 3','Assem-FRAME GAS SPRING (800N)','Assem-FRAME GAS SPRING (800N).001','Assem--FRAME KAKI')}
        class _S: pass
        d1=_S(); d1.objects=SELS[ASM]
        for ob in d1.objects: bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        import bmesh, mathutils
        kids=[ob for ob in d1.objects if ob.type=='MESH']; tot=0
        GELAP={'defaultplastic.001','defaultplastic.002','defaultplastic.004','defaultplastic.008','satinfinishaluminum','satinfinishaluminum.001','Material_5','Material_6','defaultplastic.018','defaultplastic.019','defaultplastic.020','defaultplastic.021','defaultplastic.022','defaultplastic.023','defaultplastic.012','defaultplastic.013','defaultplastic.014','defaultplastic.015','defaultplastic.016','defaultplastic.017','polishedaluminum'}   # abu-abu & silver -> hitam
        for i,ob in enumerate(kids):
            M=ob.matrix_world.copy(); ob.parent=None; ob.matrix_world=M
            bpy.context.view_layer.objects.active=ob; ob.select_set(True)
            bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
            ob.data.transform(mathutils.Matrix.Translation((float(os.environ.get('HB_DX','0')),0,0)))   # blend baru bergeser global (kerangka Kursi di x=-2.522)
            if SIDE=='Kiri': ob.data.transform(mathutils.Matrix.Translation((0,0,float(os.environ.get('HAND_DZ_KIRI','-0.03')))))   # setarakan tinggi dengan handrest kanan
            n0=len(ob.data.polygons)
            # rapikan: gabung bidang rata (CAD) dulu tanpa kehilangan bentuk, baru kurangi sisanya
            bm=bmesh.new(); bm.from_mesh(ob.data)
            bmesh.ops.dissolve_limit(bm,angle_limit=math.radians(float(os.environ.get('HAND_DISS','0.6'))),verts=bm.verts,edges=bm.edges,delimit={'MATERIAL'})
            bmesh.ops.triangulate(bm,faces=bm.faces); bm.to_mesh(ob.data); bm.free()
            ob.data.update(); n1=len(ob.data.polygons)
            tgt=int(os.environ.get('HAND_TARGET','70000'))*n0/sum(len(k.data.polygons) for k in kids) if False else None
            ob.name='Handrest_%s_%02d'%(SIDE,i); print('handrest baru',ob.name,n0,'->',n1)
        allf=sum(len(k.data.polygons) for k in kids); RATIO=min(1.0,float(os.environ.get('HAND_TARGET','120000'))/allf); print('setelah dissolve',allf,'rasio',round(RATIO,3))
        for ob in kids:
            bpy.context.view_layer.objects.active=ob
            if RATIO<1 and len(ob.data.polygons)>1500:
                md=ob.modifiers.new('dec','DECIMATE'); md.ratio=RATIO; md.delimit={'NORMAL','MATERIAL'}; md.angle_limit=math.radians(25); bpy.ops.object.modifier_apply(modifier='dec')
            me_=ob.data
            for pl in me_.polygons: pl.use_smooth=True
            try: me_.shade_smooth()
            except Exception: pass
            print('mat',ob.name,[m.name if m else None for m in me_.materials])
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
        if os.environ.get('HAND_ISI','1')=='1' and SGN!=0:
            import mathutils
            def span(o,ax): 
                vs=[v.co[ax] for v in o.data.vertices]; return max(vs)-min(vs), max(vs)
            cand=[o for o in kids if max(v.co.z for v in o.data.vertices)<-0.25]
            hs=max(cand,key=lambda o:span(o,1)[0]) if cand else None
            if hs:
                mx=(max if SGN<0 else min)(v.co.x for v in hs.data.vertices); XB=SGN*float(os.environ.get('ISI_XB','0.66')); TH=float(os.environ.get('ISI_TH','0.05'))
                pts=[v.co.copy() for v in hs.data.vertices if (v.co.x-mx)*(-SGN)>-TH][::3]
                bm=bmesh.new()
                for q in pts: bm.verts.new(q); bm.verts.new((XB,q.y,q.z))
                bmesh.ops.convex_hull(bm,input=list(bm.verts))
                for f in list(bm.faces): f.smooth=True
                hm=bpy.data.meshes.new('IsiRumahHandrest'); bm.to_mesh(hm); bm.free()
                ho=bpy.data.objects.new('Handrest_%s_Isi'%SIDE,hm); bpy.context.scene.collection.objects.link(ho)
                for pl in hm.polygons: pl.use_smooth=True
                mt=bpy.data.materials.new('isi_hitam'); mt.use_nodes=True; bs=mt.node_tree.nodes.get('Principled BSDF')
                bs.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); bs.inputs['Roughness'].default_value=0.5
                hm.materials.append(mt); print('isi rumah handrest',SIDE,hs.name,len(hm.polygons),'x',round(mx,3),'->',XB)
if HB and os.environ.get('FRAME_KAKI','1')=='1' and 'SELS' in globals():   # rangka kaki dari blend pengguna (kaki/palang di bawah kursi)
    import mathutils
    objsK=SELS['Assem--FRAME KAKI']
    for ob in objsK: bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    nk=0
    for ob in [x for x in objsK if x.type=='MESH']:
        M=ob.matrix_world.copy(); ob.parent=None; ob.matrix_world=M
        bpy.context.view_layer.objects.active=ob; ob.select_set(True)
        if ob.data.users>1: ob.data=ob.data.copy()
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        ob.data.transform(mathutils.Matrix.Translation((float(os.environ.get('HB_DX','0')),0,0)))
        for pl in ob.data.polygons: pl.use_smooth=True
        ob.name='Rangka_Kaki_%02d'%nk; nk+=1
    for ob in objsK:
        if ob.type=='EMPTY': bpy.data.objects.remove(ob)
    print('rangka kaki',nk)
if REAR_IN:
    import bmesh, mathutils
    mtR=bpy.data.materials.new('hitam_belakang'); mtR.use_nodes=True; nt=mtR.node_tree; nt.nodes.clear()
    out_=nt.nodes.new('ShaderNodeOutputMaterial'); bs_=nt.nodes.new('ShaderNodeBsdfPrincipled'); bs_.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); bs_.inputs['Roughness'].default_value=0.5; bs_.inputs['Metallic'].default_value=0.0
    nt.links.new(bs_.outputs['BSDF'],out_.inputs['Surface'])
    shells=[]
    for ASMR,SIDE in (('Assem-FRAME GAS SPRING (800N).001','Kiri'),('Assem-FRAME GAS SPRING (800N)','Kanan')):
        objsR=SELS[ASMR]
        for ob in objsR: bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        ks=[ob for ob in objsR if ob.type=='MESH']
        for ob in ks:
            M=ob.matrix_world.copy(); ob.parent=None; ob.matrix_world=M
            bpy.context.view_layer.objects.active=ob; ob.select_set(True)
            if ob.data.users>1: ob.data=ob.data.copy()
            bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
            ob.data.transform(mathutils.Matrix.Translation((float(os.environ.get('HB_DX','0')),0,0)))
            bm=bmesh.new(); bm.from_mesh(ob.data)
            bmesh.ops.dissolve_limit(bm,angle_limit=math.radians(0.6),verts=bm.verts,edges=bm.edges,delimit={'MATERIAL'})
            bmesh.ops.triangulate(bm,faces=bm.faces); bm.to_mesh(ob.data); bm.free(); ob.data.update()
        shells.append((SIDE,ks))
        for ob in objsR:
            if ob.type=='EMPTY': bpy.data.objects.remove(ob)
    allR=sum(len(k_.data.polygons) for _,ks in shells for k_ in ks); ratioR=min(1.0,float(os.environ.get('REAR_TARGET','90000'))/allR); print('cangkang belakang (blend) face',allR,'rasio',round(ratioR,3))
    for SIDE,ks in shells:
        for i,ob in enumerate(ks):
            bpy.context.view_layer.objects.active=ob
            if ratioR<1 and len(ob.data.polygons)>1500:
                md=ob.modifiers.new('dec','DECIMATE'); md.ratio=ratioR; md.delimit={'NORMAL','MATERIAL'}; md.angle_limit=math.radians(25); bpy.ops.object.modifier_apply(modifier='dec')
            me_=ob.data
            for pl in me_.polygons: pl.use_smooth=True
            me_.materials.clear(); me_.materials.append(mtR)
            ob.name='Sandaran_Belakang_%s_%02d'%(SIDE,i)
    print('cangkang belakang total face',sum(len(k_.data.polygons) for _,ks in shells for k_ in ks))
if REAR:   # cangkang belakang sandaran baru (rakitan di blend pengguna) untuk kedua jok: kiri = aslinya, kanan = salinan digeser (jok identik, tanpa cermin)
    import bmesh, mathutils
    ASMR='Assem-FRAME GAS SPRING (800N)'
    objsR=SELS[ASMR]
    for ob in objsR: bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    kidsR=[ob for ob in objsR if ob.type=='MESH']
    for i,ob in enumerate(kidsR):
        M=ob.matrix_world.copy(); ob.parent=None; ob.matrix_world=M
        bpy.context.view_layer.objects.active=ob; ob.select_set(True)
        bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
        bm=bmesh.new(); bm.from_mesh(ob.data)
        bmesh.ops.dissolve_limit(bm,angle_limit=math.radians(0.6),verts=bm.verts,edges=bm.edges,delimit={'MATERIAL'})
        bmesh.ops.triangulate(bm,faces=bm.faces); bm.to_mesh(ob.data); bm.free(); ob.data.update()
    # tempelkan: putar (sumbu x) dan geser cangkang agar sejajar & berimpit dengan area merah yang dibuang
    if os.environ.get('REAR_FIT','1')=='1' and len(RC)>100:
        big=max(kidsR,key=lambda k_:len(k_.data.polygons))
        SV=np.array([[v.co.x,v.co.y,v.co.z] for v in big.data.vertices])
        def axis(Y):
            Y=Y-Y.mean(0); u,sv,vt=np.linalg.svd(Y,full_matrices=False); d=vt[0]
            return d if d[1]>0 else -d   # arah sepanjang tinggi sandaran, (dy,dz) dgn dz>0
        def ang(P):
            d=axis(P[:,1:3]-0)
            if d[1]<0: d=-d
            return math.atan2(d[1],d[0])
        a_s=ang(SV); a_r=ang(RC); dth=a_r-a_s
        cs=SV.mean(0); cr=RC.mean(0)
        print('fit: sudut shell',round(math.degrees(a_s),1),'merah',round(math.degrees(a_r),1),'putar',round(math.degrees(dth),1),'pusat shell',np.round(cs,3),'merah',np.round(cr,3))
        dy=float(os.environ.get('REAR_DY','0.0')); dz=float(os.environ.get('REAR_DZ','0.0'))
        T=mathutils.Matrix.Translation((cr[0]-cs[0],cr[1]+dy,cr[2]+dz))@mathutils.Matrix.Rotation(dth,4,'X')@mathutils.Matrix.Translation((0,-cs[1],-cs[2]))
        for ob in kidsR: ob.data.transform(T); ob.data.update()
        # cocokkan permukaan luar cangkang ke permukaan panel merah asli (medan tinggi y(x,z)): cangkang menjorok keluar seperti panel merah, detail tetap
        if os.environ.get('REAR_RED','1')=='1':
            CS=float(os.environ.get('REAR_CELL','0.03')); xa=RC[:,0].min()-0.15; xb=RC[:,0].max()+0.15; za=RC[:,2].min()-0.15; zb_=RC[:,2].max()+0.15
            nx=int((xb-xa)/CS)+1; nz=int((zb_-za)/CS)+1
            def grid(P,fn):
                G=np.full((nx,nz),np.nan); ix=np.clip(((P[:,0]-xa)/CS).astype(int),0,nx-1); iz=np.clip(((P[:,2]-za)/CS).astype(int),0,nz-1)
                key=ix*nz+iz; o_=np.argsort(key); ks=key[o_]; cuts=np.flatnonzero(np.diff(ks))+1
                for gi in np.split(o_,cuts):
                    if len(gi)>=2: G[ix[gi[0]],iz[gi[0]]]=fn(P[gi,1])
                return G
            def fill(G,it=60):
                G=G.copy(); m=~np.isnan(G)
                for _ in range(it):
                    if m.all(): break
                    Z=np.where(m,G,0.0); W=m.astype(float)
                    Zp=np.pad(Z,1); Wp=np.pad(W,1)
                    sm=sum(Zp[1+i:1+i+nx,1+j:1+j+nz] for i in (-1,0,1) for j in (-1,0,1)); wm=sum(Wp[1+i:1+i+nx,1+j:1+j+nz] for i in (-1,0,1) for j in (-1,0,1))
                    new_=np.where(wm>0,sm/np.maximum(wm,1e-9),np.nan); G=np.where(m,G,new_); m=~np.isnan(G)
                return G
            def blur(G,it):
                for _ in range(it):
                    Gp=np.pad(G,1,mode='edge'); G=sum(Gp[1+i:1+i+nx,1+j:1+j+nz] for i in (-1,0,1) for j in (-1,0,1))/9.0
                return G
            allv=[]
            for ob in kidsR:
                a_=np.empty(len(ob.data.vertices)*3); ob.data.vertices.foreach_get('co',a_); allv.append(a_.reshape(-1,3))
            AV=np.vstack(allv)
            Gr=grid(RC,np.median); Gs=grid(AV,lambda y:np.percentile(y,97))   # y merah & permukaan belakang cangkang per sel
            D=Gr-Gs; D=fill(D); D=blur(D,int(os.environ.get('REAR_BLUR','3')))
            def samp(P):
                fx=np.clip((P[:,0]-xa)/CS-0.0,0,nx-1.001); fz=np.clip((P[:,2]-za)/CS,0,nz-1.001); i0=fx.astype(int); j0=fz.astype(int); tx=fx-i0; tz=fz-j0
                return D[i0,j0]*(1-tx)*(1-tz)+D[i0+1,j0]*tx*(1-tz)+D[i0,j0+1]*(1-tx)*tz+D[i0+1,j0+1]*tx*tz
            off=float(os.environ.get('REAR_RED_OFF','0.0'))
            for ob,a_ in zip(kidsR,allv):
                a2=a_.copy(); a2[:,1]+=samp(a2)+off; ob.data.vertices.foreach_set('co',a2.ravel()); ob.data.update()
            print('cocok merah: geser y',round(float(np.nanmin(D)),3),'..',round(float(np.nanmax(D)),3))
        # penyempurnaan: cari putaran (sumbu x) + geser y agar bagian depan cangkang menempel ke punggung badan sepanjang tinggi tanpa menembus
        if os.environ.get('REAR_REFINE','0')=='1':
            _M2=np.array(o.matrix_world); _Pb=np.empty(len(me.vertices)*3); me.vertices.foreach_get('co',_Pb); _Pb=_Pb.reshape(-1,3); _Wb=_Pb@_M2[:3,:3].T+_M2[:3,3]
            _cx0,_cx1=X0+float(os.environ.get('REAR_CXM','0.12')),X1-float(os.environ.get('REAR_CXM','0.12'))   # lebar jok kiri
            _bb=_Wb[(_Wb[:,0]>_cx0)&(_Wb[:,0]<_cx1)&(_Wb[:,2]>-0.3)&(_Wb[:,2]<0.62)]
            BIN=0.025; zb=np.arange(-0.3,0.62,BIN)
            body_rear=np.full(len(zb),np.nan)
            for i_,z_ in enumerate(zb):
                m_=(_bb[:,2]>=z_)&(_bb[:,2]<z_+BIN)
                if m_.sum()>20: body_rear[i_]=np.percentile(_bb[m_,1],99.8)
            big2=max(kidsR,key=lambda k_:len(k_.data.polygons))
            SV2=np.array([[v.co.x,v.co.y,v.co.z] for v in big2.data.vertices]); SV2=SV2[(SV2[:,0]>_cx0)&(SV2[:,0]<_cx1)]
            SV2=SV2[::max(1,len(SV2)//4000)]
            piv=np.array([SV2[:,1].mean(),SV2[:,2].min()])   # putar terhadap tepi bawah
            best=None
            for th in np.radians(np.arange(-12,12.01,0.5)):
                c_,s_=math.cos(th),math.sin(th)
                yy=(SV2[:,1]-piv[0])*c_-(SV2[:,2]-piv[1])*s_+piv[0]; zz=(SV2[:,1]-piv[0])*s_+(SV2[:,2]-piv[1])*c_+piv[1]
                idx=np.floor((zz-zb[0])/BIN).astype(int); ok=(idx>=0)&(idx<len(zb))
                front=np.full(len(zb),np.nan)
                for i_ in np.unique(idx[ok]):
                    q=yy[ok&(idx==i_)]
                    if len(q)>=5: front[i_]=np.percentile(q,5)
                gap0=front-body_rear; v_=~np.isnan(gap0)
                if v_.sum()<10: continue
                dy_=-(np.nanmin(gap0)-float(os.environ.get('REAR_TOUCH','-0.003')))   # geser y sampai celah minimum = -0.003 (menempel)
                g=gap0[v_]+dy_
                sc_=np.mean(g)
                if best is None or sc_<best[0]: best=(sc_,th,dy_,g.max())
            if best:
                sc_,th,dy_,gm=best; print('refine: putar',round(math.degrees(th),1),'geser y',round(dy_,4),'celah rata-rata',round(sc_,4),'celah maks',round(gm,4))
                c_,s_=math.cos(th),math.sin(th)
                Tr=mathutils.Matrix.Translation((0,piv[0]+dy_,piv[1]))@mathutils.Matrix.Rotation(th,4,'X')@mathutils.Matrix.Translation((0,-piv[0],-piv[1]))
                for ob in kidsR: ob.data.transform(Tr); ob.data.update()
            # lengkungkan: geser y tiap titik cangkang menurut tinggi z agar sisi depannya menempel di profil punggung badan (badan melengkung, cangkang lurus)
            if os.environ.get('REAR_WARP','0')=='1':
                allv=[]
                for ob in kidsR:
                    a_=np.empty(len(ob.data.vertices)*3); ob.data.vertices.foreach_get('co',a_); allv.append(a_.reshape(-1,3))
                AV=np.vstack(allv); AVc=AV[(AV[:,0]>_cx0)&(AV[:,0]<_cx1)]
                zc=zb+BIN/2; fr=np.full(len(zb),np.nan)
                for i_ in range(len(zb)):
                    m_=(AVc[:,2]>=zb[i_])&(AVc[:,2]<zb[i_]+BIN)
                    if m_.sum()>=5: fr[i_]=np.percentile(AVc[m_,1],3)
                dd=body_rear+float(os.environ.get('REAR_WARP_TOUCH','0.012'))-fr; ok_=~np.isnan(dd)
                if ok_.sum()>5:
                    dd=np.interp(zc,zc[ok_],dd[ok_])
                    k5=np.ones(5)/5; dp=np.pad(dd,2,mode='edge'); dd=np.convolve(dp,k5,mode='valid')
                    for ob,a_ in zip(kidsR,allv):
                        a2=a_.copy(); a2[:,1]+=np.interp(a2[:,2],zc,dd)
                        ob.data.vertices.foreach_set('co',a2.ravel()); ob.data.update()
                    print('lengkung: geser y min',round(dd.min(),3),'maks',round(dd.max(),3))
                    # buang bagian kecil yang tetap melayang jauh di belakang badan
                    okb=~np.isnan(body_rear); brz=np.interp(zc,zc[okb],body_rear[okb]); bigR=max(kidsR,key=lambda k_:len(k_.data.polygons))
                    for ob in list(kidsR):
                        if ob is bigR: continue
                        a_=np.empty(len(ob.data.vertices)*3); ob.data.vertices.foreach_get('co',a_); a_=a_.reshape(-1,3)
                        gp=float(np.min(a_[:,1]-np.interp(a_[:,2],zc,brz)))
                        if gp>float(os.environ.get('REAR_FLOAT','0.04')):
                            print('buang bagian melayang',ob.name,len(ob.data.polygons),'celah',round(gp,3)); kidsR.remove(ob); bpy.data.objects.remove(ob)
    allR=sum(len(k_.data.polygons) for k_ in kidsR); ratioR=min(1.0,float(os.environ.get('REAR_TARGET','45000'))/allR); print('cangkang belakang face',allR,'rasio',round(ratioR,3))
    mtR=bpy.data.materials.new('hitam_belakang'); mtR.use_nodes=True; nt=mtR.node_tree; nt.nodes.clear()
    out_=nt.nodes.new('ShaderNodeOutputMaterial'); bs_=nt.nodes.new('ShaderNodeBsdfPrincipled'); bs_.inputs['Base Color'].default_value=(0.025,0.025,0.028,1); bs_.inputs['Roughness'].default_value=0.5; bs_.inputs['Metallic'].default_value=0.0
    nt.links.new(bs_.outputs['BSDF'],out_.inputs['Surface'])
    for i,ob in enumerate(kidsR):
        bpy.context.view_layer.objects.active=ob
        if ratioR<1 and len(ob.data.polygons)>1500:
            md=ob.modifiers.new('dec','DECIMATE'); md.ratio=ratioR; md.delimit={'NORMAL','MATERIAL'}; md.angle_limit=math.radians(25); bpy.ops.object.modifier_apply(modifier='dec')
        me_=ob.data
        for pl in me_.polygons: pl.use_smooth=True
        me_.materials.clear(); me_.materials.append(mtR)
        ob.name='Sandaran_Belakang_Kiri_%02d'%i
    DXR=float(os.environ.get('LEVER_DX','0.827'))
    for ob in list(kidsR):
        d=ob.copy(); d.data=ob.data.copy(); d.name=ob.name.replace('Kiri','Kanan'); bpy.context.scene.collection.objects.link(d)
        d.data.transform(mathutils.Matrix.Translation((DXR,0,0)))
    for ob in objsR:
        try:
            if ob.type=='EMPTY': bpy.data.objects.remove(ob)
        except ReferenceError: pass
    print('cangkang belakang ditambahkan, total face',sum(len(k_.data.polygons) for k_ in kidsR)*2)
    # tutup celah di samping/sekeliling cangkang: tepi terbuka badan yang dekat cangkang disambung ke permukaan cangkang terdekat dengan pita halus (badan & cangkang menyatu)
    if os.environ.get('REAR_SKIRT','1')=='1':
        from mathutils.bvhtree import BVHTree
        SO=[x for x in bpy.data.objects if x.name.startswith('Sandaran_Belakang_')]
        sv=[];sp=[]
        for so in SO:
            mw=so.matrix_world; b0=len(sv); sv+= [mw@v.co for v in so.data.vertices]; sp+= [tuple(b0+i for i in p_.vertices) for p_ in so.data.polygons]
        tr=BVHTree.FromPolygons(sv,sp); DMAX=float(os.environ.get('SKIRT_DMAX','0.08')); DMIN=float(os.environ.get('SKIRT_DMIN','0.0015'))
        MWb=o.matrix_world; MWbi=MWb.inverted()
        bm=bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table(); uvl=bm.loops.layers.uv.active
        near={}
        def nr(v):
            if v.index not in near:
                h=tr.find_nearest(MWb@v.co); near[v.index]=None if h[0] is None else (h[0],h[3])
            return near[v.index]
        ne=0; pv={}; newf=[]
        for e in [e for e in bm.edges if e.is_boundary]:
            r=[nr(v) for v in e.verts]
            if None in r or any(d>DMAX for _,d in r) or all(d<DMIN for _,d in r): continue
            l=e.link_loops[0]; a=l.vert; b=l.link_loop_next.vert
            def pnt(v):
                if v.index not in pv:
                    p=nr(v)[0]; pv[v.index]=(bm.verts.new(MWbi@p), v.link_loops[0][uvl].uv.copy())
                return pv[v.index]
            pa,ua=pnt(a); pb,ub=pnt(b)
            try: f=bm.faces.new((b,a,pa,pb))
            except ValueError: continue
            f.material_index=3; f.smooth=True; ne+=1; newf.append(f)
            for lp in f.loops: lp[uvl].uv=(ua if lp.vert in (a,pa) else ub)
        bm.to_mesh(me); bm.free(); me.update(); print('pita penutup celah cangkang',ne)
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
