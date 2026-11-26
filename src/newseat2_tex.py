# Tekstur untuk seat Meshy baru: tint-base abu-abu (kulit rata + benang terang + serat halus) & normal map 2048.
# Masukan: _tmp_tex/base.png, normal.png, classmap.npy  Keluaran: _tmp_tex/tint2048.jpg, tint2048.webp, normal2048.jpg, flatuv.json
from PIL import Image
import numpy as np, json
from scipy.ndimage import gaussian_filter, binary_dilation, uniform_filter
Image.MAX_IMAGE_PIXELS=None
H=2048
base=Image.open('_tmp_tex/base.png').convert('RGB').resize((H,H),Image.BOX)
A=np.asarray(base).astype(np.float32); L=0.299*A[...,0]+0.587*A[...,1]+0.114*A[...,2]
C=np.load('_tmp_tex/classmap.npy')
thr={0:85,1:135,2:185}
S=np.zeros(L.shape,bool)
for c in range(3): S|=(C==c)&(L>thr[c])
S=binary_dilation(S,iterations=1)
LEVEL,THREAD=0.80,1.0
tint=np.zeros(L.shape,np.float32)
for c in range(3):
    sel=(C==c)&~S
    m=sel.astype(np.float32); low=gaussian_filter(np.where(sel,L,0),6)/(gaussian_filter(m,6)+1e-6)
    hp=np.where(sel,L-low,0); sd=hp[sel].std()+1e-6; hp=np.clip(hp/sd,-2.5,2.5)
    tint+=np.where(C==c,LEVEL+0.035*hp,0)
tint=np.where(S,THREAD,tint)
tint=np.clip(gaussian_filter(tint,0.7),0.5,1.0)
g=(tint*255).astype(np.uint8); rgb=np.stack([g,g,g],-1)
im=Image.fromarray(rgb); im.save('_tmp_tex/tint2048.jpg',quality=90); im.save('_tmp_tex/tint2048.webp',quality=85)
nm=Image.open('_tmp_tex/normal.png').convert('RGB').resize((H,H),Image.LANCZOS); nm.save('_tmp_tex/normal2048.jpg',quality=92)
# cari tambalan datar (untuk bantalan legrest yang tak punya UV): kelas 0, varian tint & normal terendah
N=np.asarray(nm).astype(np.float32)
w=48
v1=uniform_filter(tint,w)**1; v2=uniform_filter(tint**2,w)-uniform_filter(tint,w)**2
nv=sum(uniform_filter(N[...,k]**2,w)-uniform_filter(N[...,k],w)**2 for k in range(3))
ok=uniform_filter((C==0).astype(np.float32),w)>0.999
score=np.where(ok,v2/ (v2[ok].mean()+1e-9)+nv/(nv[ok].mean()+1e-9),1e9)
score[:64]=1e9;score[-64:]=1e9;score[:,:64]=1e9;score[:,-64:]=1e9
y,x=np.unravel_index(score.argmin(),score.shape); print('flat patch',x,y,score.min())
json.dump({'x':int(x),'y':int(y),'w':w,'H':H},open('_tmp_tex/flatuv.json','w'))
print('tint mean',tint.mean())
