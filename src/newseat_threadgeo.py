# Benang jahitan sebagai geometri 3D bersih: garis tengah dari face Tali (kontraksi titik -> jaringan garis),
# lalu dibuat pita tipis di atas permukaan kulit. Warna benang = satu material (mudah diganti).
import numpy as np, sys
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
LAB,OUT=sys.argv[1],sys.argv[2]
W=float(sys.argv[3]) if len(sys.argv)>3 else 0.003      # lebar benang (m)
OFF=float(sys.argv[4]) if len(sys.argv)>4 else 0.0      # >0 = dobel, jarak tiap garis dari garis tengah
H=0.0007                                                  # angkat dari permukaan
d=np.load('_tmp_new.npz'); cen=d['cen']; nor=d['nor']
lab=np.load(LAB); P=cen[lab==4].astype(np.float64); print('titik',len(P))
fT=cKDTree(cen)
R=0.0048; K=24
for it in range(40):
    tr=cKDTree(P); dd,ix=tr.query(P,k=K); m=dd<=R
    nb=P[ix]; w=m[...,None]; cnt=m.sum(1,keepdims=True)
    mean=(nb*w).sum(1)/np.maximum(cnt,1)
    c=nb-mean[:,None,:]; C=np.einsum('nki,nkj,nk->nij',c,c,m.astype(float))
    ev,evec=np.linalg.eigh(C); e1=evec[:,:,2]
    dl=mean-P; dl=dl-(dl*e1).sum(1,keepdims=True)*e1
    P=P+0.6*dl
# proyeksi balik ke permukaan
def snap(Q):
    _,fi=fT.query(Q); n=nor[fi]; return Q-((Q-cen[fi])*n).sum(1,keepdims=True)*n, n
P,_=snap(P)
# turunkan jumlah titik (grid) -> simpul
S=0.0024; g=np.floor(P/S).astype(np.int64); _,inv=np.unique(g,axis=0,return_inverse=True)
inv=inv.ravel(); nn=inv.max()+1
N=np.zeros((nn,3)); np.add.at(N,inv,P); N/=np.bincount(inv)[:,None]
# sambungkan simpul bertetangga
tr=cKDTree(N); pairs=np.array(sorted(tr.query_pairs(1.9*S)))
A=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(nn,nn))
nc,cl=connected_components(A,directed=False); sz=np.bincount(cl)
keepn=sz[cl]>=8
# haluskan posisi
for _ in range(3):
    s=np.zeros((nn,3)); c=np.zeros(nn)
    np.add.at(s,pairs[:,0],N[pairs[:,1]]); np.add.at(s,pairs[:,1],N[pairs[:,0]])
    np.add.at(c,pairs[:,0],1); np.add.at(c,pairs[:,1],1)
    N=np.where(c[:,None]>0,0.5*N+0.5*s/np.maximum(c,1)[:,None],N)
N,Nn=snap(N)
pairs=pairs[keepn[pairs[:,0]]&keepn[pairs[:,1]]]
print('simpul',keepn.sum(),'garis',len(pairs))
V=[];F=[];NR=[]
offs=[0.0] if OFF<=0 else [-OFF,OFF]
for a,b in pairs:
    t=N[b]-N[a]; L=np.linalg.norm(t)
    if L<1e-9: continue
    t/=L; n=Nn[a]+Nn[b]; n/=np.linalg.norm(n)+1e-12
    sd=np.cross(n,t); sd/=np.linalg.norm(sd)+1e-12
    for o in offs:
        base=sd*o
        p0=N[a]-t*W/2+base; p1=N[b]+t*W/2+base
        q=[p0-sd*W/2,p0+sd*W/2,p1+sd*W/2,p1-sd*W/2]
        i=len(V); V+= [x+n*H for x in q]; NR+=[n]*4; F+=[(i,i+1,i+2),(i,i+2,i+3)]
V=np.array(V,np.float32); F=np.array(F,np.int32); NR=np.array(NR,np.float32)
np.savez(OUT,V=V,F=F,N=NR); print('verts',len(V),'tris',len(F))
