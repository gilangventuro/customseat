# Tiap potongan benang (quad) ikut bagian terdekat (badan / handrest / footrest) supaya ikut bergerak saat dilepas.
import numpy as np, sys
from scipy.spatial import cKDTree
z=np.load(sys.argv[1]); part=np.load(sys.argv[2]); cen=np.load('_tmp_new.npz')['cen']
qc=z['V'].reshape(-1,4,3).mean(1); _,fi=cKDTree(cen).query(qc)
q=part[np.minimum(fi,len(part)-1)]; np.save(sys.argv[3],q); print('benang per bagian',np.bincount(q,minlength=5))
