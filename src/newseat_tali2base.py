# Face Tali (dipakai hanya untuk menentukan garis benang) dikembalikan ke label sekitarnya: ikut panel/kulit di kiri-kanannya.
import numpy as np, sys
from scipy.sparse import csr_matrix
d=np.load('_tmp_new.npz'); pv=d['pv']; n=len(pv)
lab=np.load(sys.argv[1]).astype(np.int8)
rows=np.repeat(np.arange(n),3); V=csr_matrix((np.ones(n*3),(rows,pv.ravel())),shape=(n,pv.max()+1))
A=(V@V.T).tocsr(); A.setdiag(0); A.eliminate_zeros()
for it in range(30):
    t=lab==4
    if not t.any(): break
    oh=np.stack([(lab==c).astype(np.float32) for c in range(4)],1)
    cnt=A@oh; has=cnt.sum(1)>0
    new=cnt.argmax(1).astype(np.int8); sel=t&has
    lab[sel]=new[sel]
print(np.bincount(lab)); np.save(sys.argv[2],lab)
