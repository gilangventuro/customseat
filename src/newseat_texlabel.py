# Rasterisasi label material per-face ke ruang UV (4096) -> peta label per-texel. Dipakai untuk menyelaraskan tekstur dgn pembagian face.
import numpy as np, sys
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
uvl=np.load('_tmp_uvl.npy'); lab=np.load(sys.argv[1]).astype(np.int8); R=4096
im=Image.new('L',(R,R),255); dr=ImageDraw.Draw(im)
# gambar urut: label 'penting' terakhir tidak perlu; tiap segitiga kecil, cukup fill langsung
P=uvl.copy(); P[...,0]=P[...,0]*R; P[...,1]=(1-P[...,1])*R
for i in range(len(lab)):
    t=P[i]; dr.polygon([(t[0,0],t[0,1]),(t[1,0],t[1,1]),(t[2,0],t[2,1])],fill=int(lab[i]))
T=np.asarray(im)
known=T!=255
# isi celah kecil: texel terdekat yang terlabel, tapi hanya dalam radius 6 px (luar itu = latar atlas)
dist,(iy,ix)=distance_transform_edt(~known,return_indices=True)
filled=T[iy,ix]; T2=np.where(dist<=6,filled,255).astype(np.uint8)
np.save('_tmp_texlabel.npy',T2); print('terlabel',(T2!=255).mean(),[ (T2==c).mean().round(3) for c in range(4)])
Image.fromarray(np.array([[40,40,40],[220,130,30],[30,200,170],[220,30,30],[255,255,255]],np.uint8)[np.minimum(T2,4)]).resize((1024,1024)).save('_tmp_texlabel.png')
