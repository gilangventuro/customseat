# Tutup lubang soket baut di kedua ujung ganggang sandaran (rev.3-HANDLE): cekungan berbentuk tetesan + lubang tembus
# diratakan mengikuti permukaan bibir di sekelilingnya. Bekerja di koordinat lokal objek (ganggang sama bentuk kiri/kanan).
# Dipanggil dari master_apply.py sebelum mesh di-bake ke dunia.
import bmesh, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# kontur luar tetesan (x,z) untuk ujung kiri (x<0), diukur dari peta tinggi; ujung kanan = cermin x
_PX=[(130,300),(135,230),(175,160),(250,125),(330,125),(450,170),(560,235),(620,305),(625,330),(600,380),(530,440),(430,490),(320,525),(265,520),(190,480),(145,400)]
_POLY=np.array([(-0.125+px/8*0.0005, 0.02-py/8*0.0005) for px,py in _PX])
# NB: peta tinggi dibuat dengan sumbu z dibalik saat digambar -> py kecil = z besar
STEP=0.0005

def _inside(poly,pts):
    x,z=pts[:,0],pts[:,1]; ins=np.zeros(len(pts),bool); n=len(poly)
    for i in range(n):
        x1,z1=poly[i]; x2,z2=poly[(i+1)%n]
        c=((z1>z)!=(z2>z))&(x<(x2-x1)*(z-z1)/(z2-z1+1e-30)+x1); ins^=c
    return ins

def _dil(m):
    g=m.copy(); g[1:]|=m[:-1]; g[:-1]|=m[1:]; g[:,1:]|=m[:,:-1]; g[:,:-1]|=m[:,1:]; return g

LF=5      # selaput bebas sampai LF sel di luar kontur; sesudahnya menempel pada permukaan asli (3 sel)
LP=9      # tambalan menutup sampai LP sel di luar kontur (1 sel = 0.5 mm)
def _cap_end(bm,bvh,side):
    poly=_POLY.copy()
    if side>0: poly[:,0]*=-1
    xs=np.arange(poly[:,0].min()-0.009,poly[:,0].max()+0.009,STEP); zs=np.arange(-0.02,0.0201,STEP)
    XX,ZZ=np.meshgrid(xs,zs); G=np.stack([XX.ravel(),ZZ.ravel()],1)
    inside=_inside(poly,G).reshape(XX.shape)
    Hh=np.full(XX.shape,np.nan)
    for i in range(len(zs)):
        for j in range(len(xs)):
            r=bvh.ray_cast(Vector((xs[j],0.06,zs[i])),Vector((0,-1,0)))
            if r[0] is not None: Hh[i,j]=r[0].y
    lev=np.full(XX.shape,99); lev[inside]=0; cur=inside.copy()
    for k in range(1,LP+1):
        nx=_dil(cur); lev[nx&~cur]=k; cur=nx
    Hb=Hh.copy(); P=np.pad(Hh,1,constant_values=np.nan); acc=np.zeros_like(Hh); cnt=np.zeros_like(Hh)
    for di in (0,1,2):
        for dj in (0,1,2):
            q=P[di:di+Hh.shape[0],dj:dj+Hh.shape[1]]; ok=~np.isnan(q); acc[ok]+=q[ok]; cnt[ok]+=1
    Hb=np.where(cnt>0,acc/np.maximum(cnt,1),np.nan); Hb[np.isnan(Hh)]=np.nan; Hh=Hb   # sedikit diperhalus (buang bintil)
    ring=(lev>LF)&(lev<=LP)&~np.isnan(Hh)
    if ring.sum()<50: print('handlecap: cincin tidak ditemukan, lewati'); return 0
    free=lev<=LF
    M=np.where(ring,Hh,np.nanmean(Hh[ring]))
    for _ in range(8000):
        N=M.copy(); N[1:-1,1:-1]=(M[:-2,1:-1]+M[2:,1:-1]+M[1:-1,:-2]+M[1:-1,2:])/4
        M=np.where(free,N,np.where(ring,Hh,M))
    def samp(x,z):
        fx=(x-xs[0])/STEP; fz=(z-zs[0])/STEP; i=int(np.clip(np.floor(fz),0,len(zs)-2)); j=int(np.clip(np.floor(fx),0,len(xs)-2)); a=fx-j; b=fz-i
        return (M[i,j]*(1-a)*(1-b)+M[i,j+1]*a*(1-b)+M[i+1,j]*(1-a)*b+M[i+1,j+1]*a*b)
    # 2) tambalan segitiga rata menutup kontur (+LP sel)
    reg=(lev<=LP)&(free|ring)
    vm={}
    def V(i,j):
        if (i,j) not in vm: vm[(i,j)]=bm.verts.new((float(xs[j]),float(M[i,j]),float(zs[i])))
        return vm[(i,j)]
    nf=0
    for i in range(len(zs)-1):
        for j in range(len(xs)-1):
            if reg[i,j] and reg[i+1,j] and reg[i+1,j+1] and reg[i,j+1]:
                a,b_,c,d=V(i,j),V(i+1,j),V(i+1,j+1),V(i,j+1)
                for t in ((a,b_,c),(a,c,d)):
                    try: bm.faces.new(t); nf+=1
                    except ValueError: pass
    print('handlecap: face tambalan',nf)
    return nf
def cap(obj):
    bm=bmesh.new(); bm.from_mesh(obj.data); bvh=BVHTree.FromBMesh(bm)
    n=0
    for side in (-1,1): n+=_cap_end(bm,bvh,side) or 0
    bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3])
    bm.to_mesh(obj.data); bm.free(); obj.data.update()
    return n
