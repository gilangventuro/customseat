# Peta kelas per-texel dari tekstur seat Meshy baru: 0 kulit utama (hitam), 1 motif/panel (cokelat), 2 list aksen (toska).
# Piksel benang (krem terang) / transisi diisi dari kelas terdekat. Keluaran: _tmp_tex/classmap.npy (2048x2048, uint8)
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
Image.MAX_IMAGE_PIXELS=None
im=Image.open('_tmp_tex/base.png').convert('RGB').resize((2048,2048),Image.BOX)
A=np.asarray(im).astype(np.float32); r,g,b=A[...,0],A[...,1],A[...,2]; L=0.299*r+0.587*g+0.114*b
teal=((g-r)>=22)&((b-r)>=12)&(L>55)
brown=((r-g)>=9)&((r-b)>=9)&(L>38)&~teal
black=(L<75)&~teal&~brown
cls=np.full(L.shape,255,np.uint8); cls[black]=0; cls[brown]=1; cls[teal]=2
unk=cls==255
print('unknown frac',unk.mean())
idx=distance_transform_edt(unk,return_distances=False,return_indices=True)
cls=cls[idx[0],idx[1]]
cls=median_filter(cls,size=5)
# buang pulau kecil (bintik kelas lain di dalam list/panel): komponen < MIN px dilebur ke kelas sekitarnya; diulang 2x
from scipy.ndimage import label
MIN=700
for _ in range(2):
    for c in range(3):
        lab,n=label(cls==c); cnt=np.bincount(lab.ravel()); small=(cnt<MIN); small[0]=False
        sm=small[lab]
        if sm.any():
            keepm=~sm; idx=distance_transform_edt(~keepm,return_distances=False,return_indices=True)
            cls=cls[idx[0],idx[1]]
# haluskan batas kelas: peta indikator di-blur lalu argmax -> tepi list mulus, tanpa gerigi/bintik
from scipy.ndimage import gaussian_filter
pr=np.stack([gaussian_filter((cls==c).astype(np.float32),6) for c in range(3)],0); cls=pr.argmax(0).astype(np.uint8)
for c in range(3):
    lab,n=label(cls==c); cnt=np.bincount(lab.ravel()); small=(cnt<MIN); small[0]=False
    if small[lab].any():
        idx=distance_transform_edt(small[lab],return_distances=False,return_indices=True); cls=cls[idx[0],idx[1]]
print('frac',[(cls==c).mean().round(3) for c in range(3)])
np.save('_tmp_tex/classmap.npy',cls)
vis=np.zeros(cls.shape+(3,),np.uint8); vis[cls==0]=(20,20,30); vis[cls==1]=(150,90,60); vis[cls==2]=(60,200,190)
Image.fromarray(vis).resize((1024,1024)).save('_tmp_tex/classmap_s.png')
