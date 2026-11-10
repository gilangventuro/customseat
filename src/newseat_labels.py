# Langkah B (python biasa): label per-face dari peta warna tekstur + aturan geometri.
# 0 main(hitam) 1 pattern(cokelat) 2 accent(hijau) 3 rest(tidak bisa diganti)
import numpy as np
from scipy.spatial import cKDTree
d=np.load('_tmp_new.npz'); cen=d['cen']; nor=d['nor']
lab=np.load('_tmp_nlab0.npy').astype(np.int8)
lab[lab==3]=0                      # merah ditangani sbg 'rest' di bawah
x,y,z=cen.T
tree=cKDTree(cen)
# normal yang dihaluskan (buang noise relief kulit)
_,nb=tree.query(cen,k=64)
ns=nor[nb].mean(1); ns/=np.linalg.norm(ns,axis=1,keepdims=True)+1e-9
nx,ny,nz=ns.T
# pattern/aksen hanya di sisi depan/atas: sisi belakang & bawah -> main
bad=(lab>0)&((ny>0.35)|(nz<-0.5))
lab[bad]=0
# aksen cuma sah kalau menempel ke panel cokelat (list di tepi panel)
dpat,_=cKDTree(cen[lab==1]).query(cen)
lab[(lab==2)&(dpat>0.06)]=0
# voting spasial supaya tepi tidak bergerigi & bintik hilang
_,nb=tree.query(cen,k=24)
for _ in range(6):
    v=lab[nb]
    cnt=np.stack([(v==c).sum(1) for c in range(3)],1)
    new=cnt.argmax(1).astype(np.int8)
    keep=cnt[np.arange(len(lab)),lab]>=cnt.max(1)   # seri -> pertahankan label lama
    lab=np.where(keep,lab,new).astype(np.int8)
    lab[(ny>0.35)|(nz<-0.5)]=np.where(lab[(ny>0.35)|(nz<-0.5)]>0,0,lab[(ny>0.35)|(nz<-0.5)])
# --- rest: bagian yang tidak boleh ikut ganti warna ---
rest=np.zeros(len(lab),bool)
rest|=(z<-0.60)&(y>-0.20)                                   # tiang/pijakan dasar
rest|=(z<-0.52)&(y>-0.36)&(y<-0.15)                         # braket footrest
arm=((np.abs(x)>0.76)|(np.abs(x)<0.07))&(z>-0.12)&(z<0.09)&(y>-0.34)&(y<0.30)
rest|=arm
dd=np.load('_tmp_nlab0.npy'); rest|=(dd==3)                 # gesper merah
lab[rest]=3
np.save('_tmp_nlab1.npy',lab)
print(np.bincount(lab))
