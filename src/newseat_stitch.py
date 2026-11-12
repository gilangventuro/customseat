# Rapikan label jahitan (4=Tali): tutup celah di sepanjang alur, buang bintik terpencil, lalu salin ke kursi sebelah
# (cermin + ICP, karena dua kursi hasil scan tidak identik persis).
import numpy as np, sys
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
SRC,OUT=sys.argv[1],sys.argv[2]
CLOSE=int(sys.argv[3]) if len(sys.argv)>3 else 2; MIN=int(sys.argv[4]) if len(sys.argv)>4 else 25
d=np.load('_tmp_new.npz'); pv=d['pv']; cen=d['cen']; n=len(pv)
lab=np.load(SRC).astype(np.int8)
# adjacency via titik sudut bersama
rows=np.repeat(np.arange(n),3); V=csr_matrix((np.ones(n*3),(rows,pv.ravel())),shape=(n,pv.max()+1))
A=(V@V.T).tocsr(); A.data[:]=1; A.setdiag(0); A.eliminate_zeros()
def dil(x,k=1):
    x=x.astype(np.float32)
    for _ in range(k): x=(A@x+x)>0
    return x.astype(bool)
def ero(x,k=1): return ~dil(~x,k)
def tidy(T,allowed):
    C=ero(dil(T,CLOSE)&allowed,CLOSE)|T   # closing (tutup celah), hanya melebar ke label yg boleh
    # buang pulau kecil
    G=A[C][:,C]; nc,cl=connected_components(G,directed=False); sz=np.bincount(cl)
    keep=np.zeros(n,bool); idx=np.where(C)[0]; keep[idx]=sz[cl]>=MIN
    return keep
T=lab==4
side_r=cen[:,0]>0
allowed=(lab==0)|(lab==1)|T                   # jangan menimpa list aksen / bagian tak-berubah
Tr=tidy(T&side_r,allowed&side_r)
print('kanan sebelum',(T&side_r).sum(),'sesudah',Tr.sum())
# --- salin ke kursi kiri: cermin x + ICP rigid ---
L=np.where(~side_r&(np.abs(cen[:,0])>0.02))[0]; Rf=np.where(side_r&(np.abs(cen[:,0])>0.02))[0]
tl=cKDTree(cen[L]); P=cen[Rf]*[-1,1,1]; Pt=P.copy()
for it in range(25):
    dd,idx=tl.query(Pt); sel=dd<np.percentile(dd,80); a,b=Pt[sel],cen[L][idx[sel]]
    ca,cb=a.mean(0),b.mean(0); U,S,Vt=np.linalg.svd((a-ca).T@(b-cb)); r=Vt.T@U.T
    if np.linalg.det(r)<0: Vt[-1]*=-1; r=Vt.T@U.T
    Pt=(Pt-ca)@r.T+cb
# penyelarasan lokal per wilayah (scan dua kursi tidak identik): ICP kecil per klaster titik Tali
from scipy.cluster.vq import kmeans2
ids=np.where(Tr[Rf])[0]; q=Pt[ids].copy()
Pall=Pt; K=24
_,km=kmeans2(q,K,seed=1,minit='++'); q2=q.copy(); trR=cKDTree(Pall)
for c in range(K):
    m=km==c
    if m.sum()<5: continue
    ctr=q[m].mean(0); near=trR.query_ball_point(ctr,0.12); X=Pall[near]
    R_=np.eye(3); t_=np.zeros(3); Y=X.copy()
    for it in range(15):
        dd,ix=tl.query(Y); sel=dd<np.percentile(dd,85); a,b=Y[sel],cen[L][ix[sel]]
        ca,cb=a.mean(0),b.mean(0); U,S,Vt=np.linalg.svd((a-ca).T@(b-cb)); r=Vt.T@U.T
        if np.linalg.det(r)<0: Vt[-1]*=-1; r=Vt.T@U.T
        Y=(Y-ca)@r.T+cb; R_=r@R_; t_=(t_-ca)@r.T+cb
    q2[m]=(q[m]-X.mean(0)) if False else q[m]
    # terapkan transformasi akumulasi: hitung dari pasangan X->Y (Kabsch)
    ca,cb=X.mean(0),Y.mean(0); U,S,Vt=np.linalg.svd((X-ca).T@(Y-cb)); r=Vt.T@U.T
    if np.linalg.det(r)<0: Vt[-1]*=-1; r=Vt.T@U.T
    q2[m]=(q[m]-ca)@r.T+cb
dd,idx=tl.query(q2); ok=dd<0.004
for nm,msk in (('footrest',cen[Rf][ids,2]<-0.45),('seat',(cen[Rf][ids,2]>=-0.45)&(cen[Rf][ids,2]<-0.05)),('back',cen[Rf][ids,2]>=-0.05)):
    print('residu',nm,np.median(dd[msk]).round(4),np.percentile(dd[msk],90).round(4),'ok',ok[msk].mean().round(2))
Tl=np.zeros(n,bool); Tl[L[idx[ok]]]=True
Tl&=(lab==0)|(lab==1)|(lab==4)
# isi sela antar titik hasil salinan: closing + buang pulau lagi
Tl=tidy(Tl,((lab==0)|(lab==1)|(lab==4))&~side_r)&~side_r
print('kiri',Tl.sum(),'(tersalin',ok.sum(),'dari',len(ids),')')
new=lab.copy(); new[lab==4]=0; new[(Tr|Tl)]=4
# label asli selain 0/1 tidak ditimpa
new[(Tr|Tl)&~((lab==0)|(lab==1)|(lab==4))]=lab[(Tr|Tl)&~((lab==0)|(lab==1)|(lab==4))]
print(np.bincount(new)); np.save(OUT,new)
