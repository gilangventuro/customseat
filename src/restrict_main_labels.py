"""Batasi leather_main (Kulit Utama) hanya ke headrest, bolster sandaran/dudukan, sisi footrest.
Bagian lain (belakang sandaran bawah, kolong, pedestal, sandaran tangan) jadi material leather_rest
(tidak bisa diwarnai). Pola berlian diperluas ke seluruh panel (isi lubang) lewat restrict_main_fill.py.
Urutan: blender --background --python src/restrict_main_dump.py ; python src/restrict_main_labels.py ;
        blender --background --python src/restrict_main_apply.py -- assets/twin-seat-meshy.glb  (jalankan dari root proyek)
"""
import numpy as np, sys
from scipy.spatial import cKDTree
d=np.load('_tmp_faces.npz');c=d['c'];n=d['n'];m=d['m']
base=np.zeros(len(c),dtype=np.int8)
base[m=='leather_pattern']=1; base[m=='leather_accent']=2
lab=base.copy()
ZH=float(sys.argv[1]) if len(sys.argv)>1 else 0.092
tree0=cKDTree(c); _,i0=tree0.query(c,k=96)
same0=np.einsum('ij,ikj->ik',n,n[i0])>0.2
ns=(n[i0]*same0[...,None]).sum(1); ns/=np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-9)
np.save('_tmp_ns.npy',ns)
nx,ny,nz=ns[:,0],ns[:,1],ns[:,2]
# pattern hanya di permukaan hadap depan/atas, bukan sisi/bawah/belakang
bad_pat=(lab==1)&((np.abs(nx)>0.7)|(nz<-0.3)|(ny>0.3))
lab[bad_pat]=0
# bagian di luar zona "kulit utama": belakang sandaran (di bawah headrest) & bawah
rest=(lab==0)&(((ny>0.4)&(np.abs(nx)<0.65)&(c[:,2]<ZH))|(nz<-0.4))
lab[rest]=3
x,y,z=c[:,0],c[:,1],c[:,2]
# dudukan/pedestal/kaki & sandaran tangan bukan kulit utama
ped=(z<0.024)&(y>-0.02)
arm=(z>0.054)&(z<0.068)&(y>-0.026)&(y<0.012)&((np.abs(x)>0.052)|(np.abs(x)<0.006))
lab[(lab==0)&(ped|arm)]=3
# smoothing spasial (tahap 1)
tree=cKDTree(c); _,idx=tree.query(c,k=32)
same=np.einsum('ij,ikj->ik',n,n[idx])>0
for p in range(8):
    nl=lab[idx]; onehot=np.stack([((nl==k)&same).sum(1) for k in range(4)],1)
    tot=onehot.sum(1); top=onehot.argmax(1)
    ch=(top!=lab)&(onehot[np.arange(len(lab)),top]/np.maximum(tot,1)>=0.5)&(tot>=8)
    lab[ch]=top[ch]; print('pass',p,ch.sum())
from restrict_main_fill import fill_panels
lab=fill_panels(c,ns,lab)
np.save('_tmp_lab.npy',lab)
print([(k,(lab==k).sum()) for k in range(4)])
