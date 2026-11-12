# Gambar ulang jahitan sebagai benang bersih di ruang UV: garis tengah (skeleton) dari tandaan Tali -> garis tipis rata, anti-alias.
# Output: RGBA 4096 (benang putih, alpha = ketebalan) untuk dipasang di mesh overlay.
import numpy as np, sys
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_closing, binary_fill_holes, label, binary_dilation, convolve
from skimage.morphology import skeletonize
TL,OUT=sys.argv[1],sys.argv[2]
W=float(sys.argv[3]) if len(sys.argv)>3 else 2.6       # tebal benang (px @4096)
OFF=float(sys.argv[4]) if len(sys.argv)>4 else 0.0     # >0: jahitan dobel, jarak tiap garis dari garis tengah
SPUR=int(sys.argv[5]) if len(sys.argv)>5 else 14
T=np.load(TL); m=T==4
m=binary_closing(m,iterations=2); m=binary_fill_holes(m)
lb,n=label(m,structure=np.ones((3,3))); sz=np.bincount(lb.ravel()); m=(sz[lb]>=60)&(lb>0)
sk=skeletonize(m)
# buang cabang pendek (spur): hapus ujung berulang, lalu tumbuhkan kembali hanya ujung lama yang bukan spur
K=np.ones((3,3)); 
def ends(s): return s&(convolve(s.astype(int),K,mode='constant')==2)
s=sk.copy(); removed=[]
for i in range(SPUR):
    e=ends(s); removed.append(e); s=s&~e
# regrow: ujung yang tersisa tumbuh kembali sepanjang jejak yang dihapus (agar garis utama tidak memendek)
cur=s
for i in range(SPUR-1,-1,-1):
    e=ends(cur); g=binary_dilation(e,structure=K)&removed[i]
    cur=cur|g
sk=cur
d=distance_transform_edt(~sk)
def ln(d0): return np.clip(W/2+0.5-np.abs(d-d0),0,1)
a=ln(0) if OFF<=0 else np.maximum(ln(OFF),ln(-OFF)) if False else np.maximum(np.clip(W/2+0.5-np.abs(d-OFF),0,1),0)
a=a*binary_dilation(m,iterations=int(OFF+3)).astype(float)
img=np.zeros((*a.shape,4),np.uint8); img[...,:3]=255; img[...,3]=(a*255).astype(np.uint8)
Image.fromarray(img).save(OUT)
print('skeleton px',sk.sum(),'alpha cover',(a>0.5).mean())
