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

def _cap_end(bm,bvh,side):
    poly=_POLY.copy()
    if side>0: poly[:,0]*=-1
    xs=np.arange(poly[:,0].min()-0.004,poly[:,0].max()+0.004,STEP); zs=np.arange(-0.02,0.0201,STEP)
    XX,ZZ=np.meshgrid(xs,zs); G=np.stack([XX.ravel(),ZZ.ravel()],1)
    inside=_inside(poly,G).reshape(XX.shape)
    # peta tinggi permukaan luar (sinar dari +y ke -y)
    Hh=np.full(XX.shape,np.nan)
    for i in range(len(zs)):
        for j in range(len(xs)):
            r=bvh.ray_cast(Vector((xs[j],0.06,zs[i])),Vector((0,-1,0)))
            if r[0] is not None: Hh[i,j]=r[0].y
    # cincin di luar kontur (3 sel) = batas Dirichlet; selaput Laplace di dalam
    grow=inside.copy()
    for _ in range(3):
        g=grow.copy(); g[1:]|=grow[:-1]; g[:-1]|=grow[1:]; g[:,1:]|=grow[:,:-1]; g[:,:-1]|=grow[:,1:]; grow=g
    ring=grow&~inside&~np.isnan(Hh)
    if ring.sum()<20: print('handlecap: cincin tidak ditemukan, lewati'); return 0
    M=np.where(ring,Hh,np.nanmean(Hh[ring]))
    for _ in range(3000):
        N=M.copy(); N[1:-1,1:-1]=(M[:-2,1:-1]+M[2:,1:-1]+M[1:-1,:-2]+M[1:-1,2:])/4
        M=np.where(inside,N,np.where(ring,Hh,M))
    def samp(x,z):
        fx=(x-xs[0])/STEP; fz=(z-zs[0])/STEP; i=int(np.clip(np.floor(fz),0,len(zs)-2)); j=int(np.clip(np.floor(fx),0,len(xs)-2)); a=fx-j; b=fz-i
        return (M[i,j]*(1-a)*(1-b)+M[i,j+1]*a*(1-b)+M[i+1,j]*(1-a)*b+M[i+1,j+1]*a*b)
    # tambalan: jaring segitiga rata (mengikuti selaput) menutup seluruh kontur; cekungan lama tertutup di bawahnya
    reg=inside.copy(); g=reg.copy(); g[1:]|=reg[:-1]; g[:-1]|=reg[1:]; g[:,1:]|=reg[:,:-1]; g[:,:-1]|=reg[:,1:]; reg=g&(inside|ring)
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
