# Tandai jahitan pola tegas (garis alur ganda pada panel kulit) sebagai material tersendiri -> warnanya bisa diganti di situs.
# Model hasil scan tidak punya material untuk benang jahitan; alur jahitan hanya berupa lekukan sempit pada mesh.
# Deteksi: lekukan sempit = titik yang tertarik ke dalam relatif terhadap permukaan yang dihaluskan sedikit vs dihaluskan lebih banyak
# (band-pass kelengkungan), lalu bercak lebar (cekungan, bukan garis) dibuang.
import numpy as np

def _ops(me):
    n=len(me.vertices)
    E=np.empty(len(me.edges)*2,np.int64); me.edges.foreach_get('vertices',E); E=E.reshape(-1,2)
    deg=np.maximum(np.bincount(E.ravel(),minlength=n).astype(np.float64),1)
    a,b=E[:,0],E[:,1]
    def lap(X):
        X=np.asarray(X,np.float64); one=X.ndim==1
        if one: X=X[:,None]
        out=np.empty_like(X)
        for c in range(X.shape[1]):
            out[:,c]=np.bincount(a,X[b,c],minlength=n)+np.bincount(b,X[a,c],minlength=n)
        out/=deg[:,None]
        return out[:,0] if one else out
    return n,lap

def stitch_face_mask(me,it_small=2,it_large=10,pct=16.0,blob_rings=4,dilate=1):
    n,lap=_ops(me)
    V=np.empty(n*3); me.vertices.foreach_get('co',V); V=V.reshape(-1,3)
    N=np.empty(n*3); me.vertices.foreach_get('normal',N); N=N.reshape(-1,3)
    S=V.copy(); Ss=None
    for i in range(it_large):
        S=0.5*S+0.5*lap(S)
        if i+1==it_small: Ss=S.copy()
    d=((Ss-S)*N).sum(1)
    sel=d<np.percentile(d,pct)
    # buang bercak lebar (cekungan): inti = titik yang semua tetangganya ikut terpilih, lalu dikembangkan beberapa lingkar
    core=sel&(lap(sel.astype(np.float64))>0.999)
    blob=core.copy()
    for _ in range(blob_rings): blob=(lap(blob.astype(np.float64))>0)&sel|blob
    sel&=~blob
    for _ in range(dilate): sel=(lap(sel.astype(np.float64))>0)|sel
    L=np.empty(len(me.loops),np.int64); me.loops.foreach_get('vertex_index',L)
    st=np.empty(len(me.polygons),np.int64); me.polygons.foreach_get('loop_start',st)
    tot=np.empty(len(me.polygons),np.int64); me.polygons.foreach_get('loop_total',tot)
    cnt=np.add.reduceat(sel[L].astype(np.int64),st)
    return cnt==tot

def mark_stitch(o,mk,matdef,**kw):
    """o: objek kulit (sudah diremap ke leather_*). Faces alur jahitan -> material leather_stitch (kecuali list aksen)."""
    me=o.data
    names=[m.name for m in me.materials]
    if 'leather_stitch' not in names: me.materials.append(mk('leather_stitch',*matdef['leather_stitch'])); names.append('leather_stitch')
    si=names.index('leather_stitch'); ai=names.index('leather_accent') if 'leather_accent' in names else -1
    mask=stitch_face_mask(me,**kw)
    mi=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('material_index',mi)
    mask&=(mi!=ai)
    mi[mask]=si; me.polygons.foreach_set('material_index',mi); me.update()
    return int(mask.sum())
