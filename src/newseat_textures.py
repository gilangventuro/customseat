# Langkah C: tekstur 4096 (base jpg), normal 2048 (jpg), dan tint-base (webp) yang dinormalisasi per kelas warna
from PIL import Image, ImageFilter
import numpy as np, sys, os
Image.MAX_IMAGE_PIXELS=None
d=sys.argv[1]; pre=sys.argv[2]; out=sys.argv[3]
base=Image.open(f'{d}/{pre}.png').convert('RGB')
b4=base.resize((4096,4096),Image.LANCZOS)
# warna default: panel cokelat jadi hitam (detail jahitan tetap), langsung di-bake ke tekstur
A=np.asarray(b4).astype(float); r,g,b=A[...,0],A[...,1],A[...,2]
msk=(((r-b)>=6)&(r>g)).astype(np.uint8)*255
msk=np.asarray(Image.fromarray(msk).filter(ImageFilter.GaussianBlur(2))).astype(float)/255
Lm=0.299*r+0.587*g+0.114*b
gray=np.stack([Lm*0.72,Lm*0.70,Lm*0.71],-1)
A=A*(1-msk[...,None])+gray*msk[...,None]
b4=Image.fromarray(np.clip(A,0,255).astype(np.uint8))
b4.save(f'{out}/_tmp_base4k.jpg',quality=88)
nm=Image.open(f'{d}/{pre}_normal.png').convert('RGB').resize((2048,2048),Image.LANCZOS)
nm.save(f'{out}/_tmp_normal2k.jpg',quality=88)
lm=np.load('_tmp_labmap.npy')               # 2048
R=2048
a=np.asarray(base.resize((R,R),Image.BOX)).astype(float)
L=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
# bobot kelas halus
W=[]
for c in range(4):
    m=Image.fromarray(((lm==c)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    W.append(np.asarray(m).astype(float)/255)
W=np.stack(W,0); W/=W.sum(0,keepdims=True)+1e-6
tint=np.zeros((R,R))
for c in range(4):
    sel=lm==c
    p50=np.percentile(L[sel],50); sd=L[sel].std()+1e-6
    t=0.86+0.10*(L-p50)/sd
    tint+=W[c]*np.clip(t,0.40,1.0)
t4=Image.fromarray(np.clip(tint*255,0,255).astype(np.uint8)).resize((4096,4096),Image.BICUBIC).convert('RGB')
t4.save(f'{out}/_tmp_tint4k.webp',quality=82)
print('ok',os.path.getsize(f'{out}/_tmp_base4k.jpg')/1e6,os.path.getsize(f'{out}/_tmp_normal2k.jpg')/1e6,os.path.getsize(f'{out}/_tmp_tint4k.webp')/1e6)
