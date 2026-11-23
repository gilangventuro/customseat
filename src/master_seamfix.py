# Perbaikan lokal badan twin seat: garis jahitan (alur pembatas) di ujung blok kulit kanan-depan putus ~9 cm
# (mesh Meshy-nya rusak/berantakan di situ). Alur dibangun ulang mengikuti lintasan antara dua ujung yang masih utuh,
# memakai profil alur yang diukur dari bagian jahitan yang utuh. Koordinat = dunia Blender (mesh sudah di-bake).
# Dipanggil dari master_apply.py sebelum decimate. Idempotent terhadap master lain: kalau geometri di titik itu tidak ada, tidak berbuat apa-apa.
import bmesh, numpy as np, mathutils

# titik-titik lintasan alur (dunia Blender, dari atas ke bawah): ujung atas yang utuh, tengah (di dekat ujung bawah celah), dan bawah.
# Alur di bagian bawah memang sudah samar di master; ikut diperjelas supaya garisnya menerus sampai bawah blok.
PTS=np.array([(1.1443,-1.9005,0.9700),(1.1367,-1.9092,0.9201),(1.1300,-1.8968,0.8315),(1.1028,-1.8747,0.7455)])
TS=np.array([(-0.19,-0.26,-1.0),(-0.237,0.18,-1.0)])   # arah singgung ujung atas & bawah (turun)
HALF=0.035      # setengah lebar pita yang diproses (m)
DEPTH=0.0065    # kedalaman alur terhadap permukaan rata (m); alur utuh di sekitarnya 2.5-8 mm
SIGMA=0.0046    # lebar Gaussian alur (m)
OFF0=0.0225; OFF1=0.0165   # jarak alur kedua (jahitan ganda) dari alur utama: di ujung atas -> ujung bawah (m)
BEAD=0.0045; DEPTH2=0.0042; SIGMA2=0.0038
MAXEDGE=0.0025  # rapatkan mesh di pita sampai sisi terpanjang <= ini

def _path(n=120):
    P=PTS; T=[TS[0]/np.linalg.norm(TS[0])*np.linalg.norm(P[1]-P[0])]
    for i in range(1,len(P)-1): T.append((P[i+1]-P[i-1])/2)
    T.append(TS[1]/np.linalg.norm(TS[1])*np.linalg.norm(P[-1]-P[-2]))
    out=[]
    for i in range(len(P)-1):
        t=np.linspace(0,1,n//(len(P)-1),endpoint=(i==len(P)-2))[:,None]
        h00=2*t**3-3*t**2+1; h10=t**3-2*t**2+t; h01=-2*t**3+3*t**2; h11=t**3-t**2
        out.append(h00*P[i]+h10*T[i]+h01*P[i+1]+h11*T[i+1])
    return np.concatenate(out)

def _band(P,path,extra=0.0):
    """untuk tiap titik: (jarak ke lintasan, parameter t di lintasan, titik terdekat)"""
    seg=path[1:]-path[:-1]; sl=(seg**2).sum(1)
    best=np.full(len(P),9.0); bt=np.zeros(len(P)); bi=np.zeros(len(P),int)
    for i in range(len(seg)):
        w=np.clip(((P-path[i])@seg[i])/sl[i],0,1); q=path[i]+w[:,None]*seg[i]
        d=np.linalg.norm(P-q,axis=1); m=d<best; best[m]=d[m]; bt[m]=(i+w[m])/len(seg); bi[m]=i
    return best,bt,bi

def repair(body):
    me=body.data; bm=bmesh.new(); bm.from_mesh(me)
    path=_path()
    co=np.array([v.co[:] for v in bm.verts]); d,t,_=_band(co,path)
    if (d<HALF).sum()<50: print('seamfix: geometri tidak ditemukan, lewati'); bm.free(); return
    # 1) rapatkan mesh di pita
    for _ in range(4):
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        co=np.array([v.co[:] for v in bm.verts]); d,t,_=_band(co,path)
        inb=(d<(HALF+OFF0)*1.2)&(t>-0.02)&(t<1.02)
        ed=[e for e in bm.edges if (inb[e.verts[0].index] or inb[e.verts[1].index]) and e.calc_length()>MAXEDGE]
        if not ed: break
        bmesh.ops.subdivide_edges(bm,edges=ed,cuts=1,use_grid_fill=True)
    bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3])
    bm.verts.ensure_lookup_table(); bm.normal_update()
    co=np.array([v.co[:] for v in bm.verts]); nrm=np.array([v.normal[:] for v in bm.verts])
    d,t,bi=_band(co,path);
    reg=np.where(d<0.05)[0]; regset=set(reg.tolist())
    ns=nrm.copy()
    nbr={int(v):[u.index for e in bm.verts[int(v)].link_edges for u in [e.other_vert(bm.verts[int(v)])]] for v in reg}
    for _ in range(14):
        prev=ns.copy()
        for v in reg: ns[v]=prev[nbr[int(v)]+[int(v)]].mean(0)
        ns/=np.linalg.norm(ns,axis=1)[:,None]+1e-12
    nrm_raw=nrm; nrm=ns   # normal dihaluskan: alur tidak ikut bergelombang oleh noise mesh asli
    # lateral = jarak ke lintasan pada bidang singgung permukaan
    seg=path[1:]-path[:-1]; tang=seg[bi]/np.linalg.norm(seg[bi],axis=1)[:,None]
    q=path[bi]+np.clip(((co-path[bi])*seg[bi]).sum(1)/(seg[bi]**2).sum(1),0,1)[:,None]*seg[bi]
    r=co-q; r=r-(r*nrm).sum(1)[:,None]*nrm
    ld=np.cross(-tang,nrm); ld/=np.linalg.norm(ld,axis=1)[:,None]+1e-12
    sg=(r*ld).sum(1); off=np.interp(t,[0,0.2,1],[OFF0,0.018,OFF1])           # sg>0 = sisi tempat alur kedua
    lat=np.abs(sg); lat2=np.abs(sg-off); dseg=np.maximum(0,np.maximum(-sg,sg-off))
    inb=np.where((dseg<HALF)&(t>0.0)&(t<1.0))[0]
    # bobot sepanjang lintasan (0 di ujung utuh, 1 di tengah) x bobot melintang (1 di alur, 0 di tepi pita)
    sm=lambda x: x*x*(3-2*x)
    wl=sm(np.clip(t/0.2,0,1))*sm(np.clip((1-t)/0.04,0,1))
    wc=sm(np.clip((HALF-dseg)/(HALF*0.5),0,1))
    # permukaan terlipat (face terbalik / serpihan) di sekitar lintasan: vertexnya ikut diratakan penuh
    nb_all=[[u.index for e in v.link_edges for u in [e.other_vert(v)]] for v in bm.verts] if False else None
    fix=set()
    for v in reg:
        for f in bm.verts[v].link_faces:
            if f.normal.dot(mathutils.Vector(ns[[x.index for x in f.verts]].mean(0)))<0.5:
                for x in f.verts:
                    fix.add(x.index)
                    for e in x.link_edges: fix.add(e.other_vert(x).index)
    fix=np.array(sorted(fix&regset),int); print('seamfix: vertex terlipat',len(fix))
    wfull=wl*wc; wfull[fix]=1.0
    inb=np.union1d(inb,fix); w=wfull[inb]
    # 2) ratakan kerusakan: Laplacian terbatas pada pita (bobot w)
    idx={int(v):i for i,v in enumerate(inb)}
    nb=[[u.index for e in bm.verts[v].link_edges for u in [e.other_vert(bm.verts[v])]] for v in inb]
    Q=co.copy()
    for _ in range(100):
        N=Q.copy()
        for i,v in enumerate(inb): N[v]=Q[v]*(1-w[i])+Q[nb[i]].mean(0)*w[i]
        Q=N
    # 3) alur: Gaussian ke arah dalam (melawan normal), penuh di lintasan
    # posisi lateral dihitung ulang dari titik yang sudah dihaluskan (Q), supaya tepi alur tidak bergelombang oleh noise mesh asli
    dq,tq,bq=_band(Q[inb],path); sgm=seg[bq]; tgq=sgm/np.linalg.norm(sgm,axis=1)[:,None]
    qq=path[bq]+np.clip(((Q[inb]-path[bq])*sgm).sum(1)/(sgm**2).sum(1),0,1)[:,None]*sgm
    rq=Q[inb]-qq; rq=rq-(rq*nrm[inb]).sum(1)[:,None]*nrm[inb]
    ldq=np.cross(-tgq,nrm[inb]); ldq/=np.linalg.norm(ldq,axis=1)[:,None]+1e-12
    sgq=(rq*ldq).sum(1); offq=np.interp(tq,[0,0.2,1],[OFF0,0.018,OFF1])
    lat=lat.copy(); lat2=lat2.copy(); sg=sg.copy(); off=off.copy()
    sg[inb]=sgq; off[inb]=offq; lat[inb]=np.abs(sgq); lat2[inb]=np.abs(sgq-offq)
    prof=(DEPTH*np.exp(-(lat[inb]/SIGMA)**2)+DEPTH2*np.exp(-(lat2[inb]/SIGMA2)**2)-BEAD*np.exp(-((sg[inb]-off[inb]/2)/(off[inb]*0.3))**2))*wl[inb]
    new=Q[inb]-nrm[inb]*prof[:,None]
    for i,v in enumerate(inb): bm.verts[v].co=mathutils.Vector(new[i])
    # 4) garis jahitan: face di dasar alur diberi material benang (seperti di bagian jahitan yang utuh)
    sidx=next((i for i,m in enumerate(me.materials) if m and m.name=='defaultplastic.001'),None)
    if sidx is not None:
        near=np.zeros(len(bm.verts),bool); near[inb[((lat[inb]<0.0022)|(lat2[inb]<0.0022))&(wl[inb]>0.5)]]=True
        for f in bm.faces:
            if all(near[v.index] for v in f.verts): f.material_index=sidx
    print('seamfix: pita',len(inb),'vertex; total vertex',len(bm.verts))
    bm.to_mesh(me); bm.free(); me.update()


def despike(body,thr=0.25,k=0.8,iters=5):
    """Duri/bintil kecil di sepanjang jahitan (noise mesh Meshy): titik yang menyimpang dari rata-rata tetangganya
    melawan normalnya lebih dari thr x panjang sisi rata-rata digeser k bagian ke rata-rata itu. Titik lain tidak disentuh."""
    me=body.data; n=len(me.vertices)
    co=np.empty(n*3); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
    ed=np.empty(len(me.edges)*2,int); me.edges.foreach_get('vertices',ed); ed=ed.reshape(-1,2)
    tot=0
    for _ in range(iters):
        nr=np.empty(n*3); me.vertices.foreach_get('normal',nr); nr=nr.reshape(-1,3)
        a,b=ed[:,0],ed[:,1]
        acc=np.zeros((n,3)); cnt=np.zeros(n); el=np.zeros(n)
        np.add.at(acc,a,co[b]); np.add.at(acc,b,co[a]); np.add.at(cnt,a,1); np.add.at(cnt,b,1)
        L=np.linalg.norm(co[a]-co[b],axis=1); np.add.at(el,a,L); np.add.at(el,b,L)
        cen=acc/np.maximum(cnt,1)[:,None]; el/=np.maximum(cnt,1)
        dv=cen-co; s=(dv*nr).sum(1)
        sp=(np.abs(s)>thr*el)&(cnt>2)
        co[sp]+=dv[sp]*k; tot+=int(sp.sum())
        me.vertices.foreach_set('co',co.ravel()); me.update()
    print('despike: titik duri dihaluskan',tot)


def clean_thread(body,minf=30):
    """Bercak/serpihan kecil bermaterial benang (hasil scan) di sekitar jahitan tampak seperti kotoran pada warna terang.
    Kumpulan face benang < minf face dikembalikan ke material kulit tetangganya; garis jahitan yang utuh tidak disentuh."""
    me=body.data; bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    th={i for i,m in enumerate(me.materials) if m and m.name in ('defaultplastic.001','Jahitan')}
    isth=[f.material_index in th for f in bm.faces]
    seen=[False]*len(bm.faces); fixed=0; comps=0
    for f in bm.faces:
        if not isth[f.index] or seen[f.index]: continue
        comp=[f]; seen[f.index]=True; k=0
        while k<len(comp):
            for v in comp[k].verts:
                for g in v.link_faces:
                    if isth[g.index] and not seen[g.index]: seen[g.index]=True; comp.append(g)
            k+=1
        comps+=1
        if len(comp)>=minf: continue
        cnt={}
        for x in comp:
            for v in x.verts:
                for g in v.link_faces:
                    if not isth[g.index]: cnt[g.material_index]=cnt.get(g.material_index,0)+1
        if not cnt: continue
        mi=max(cnt,key=cnt.get)
        for x in comp: x.material_index=mi
        fixed+=len(comp)
    print('clean_thread: komponen',comps,'face benang kecil dikembalikan ke kulit',fixed)
    bm.to_mesh(me); bm.free(); me.update()
