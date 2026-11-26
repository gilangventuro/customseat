# Tekstur untuk seat Meshy baru: tint-base abu-abu (kulit rata + benang terang + serat halus) & normal map 2048.
# Masukan: _tmp_tex/base.png, normal.png, classmap.npy  Keluaran: _tmp_tex/tint2048.jpg, tint2048.webp, normal2048.jpg, flatuv.json
from PIL import Image
import numpy as np, json, os
from scipy.ndimage import gaussian_filter, binary_dilation, uniform_filter
Image.MAX_IMAGE_PIXELS=None
H=2048
base=Image.open('_tmp_tex/base.png').convert('RGB').resize((H,H),Image.BOX)
A=np.asarray(base).astype(np.float32); L=0.299*A[...,0]+0.587*A[...,1]+0.114*A[...,2]
C=np.load('_tmp_tex/classmap.npy')
# benang: deret manik terang di alur jahit (4096 px) -> manik asli apa adanya (white top-hat), bercak <6 px dibuang.
# Manik ditanam langsung di tint-base (nilai 1.0 vs kulit 0.80); shader situs mewarnai piksel terang itu dengan warna benang.
from scipy.ndimage import grey_opening, label
B4=np.asarray(Image.open('_tmp_tex/base.png').convert('RGB')).astype(np.float32); L4=0.299*B4[...,0]+0.587*B4[...,1]+0.114*B4[...,2]
beads=(L4-grey_opening(L4,size=(9,9)))>float(os.environ.get('BEAD_T','28'))
lab,n=label(beads); cnt=np.bincount(lab.ravel()); keepc=cnt>=6; keepc[0]=False; beads=keepc[lab]
beads=beads
Sf=np.asarray(Image.fromarray((beads*255).astype(np.uint8)).resize((H,H),Image.BOX)).astype(np.float32)/255
S=Sf>0.3
LEVEL,THREAD=0.80,0.96
tint=np.zeros(L.shape,np.float32)
for c in range(3):
    sel=(C==c)&~S
    m=sel.astype(np.float32); low=gaussian_filter(np.where(sel,L,0),6)/(gaussian_filter(m,6)+1e-6)
    hp=np.where(sel,L-low,0); sd=hp[sel].std()+1e-6; hp=np.clip(hp/sd,-2.5,2.5)
    tint+=np.where(C==c,LEVEL+0.035*hp,0)
tint=tint+(1.0-tint)*np.clip(Sf*2.4,0,1)
tint=np.clip(tint,0.5,1.0)
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
# albedo asli (2048) untuk tampilan bawaan = seperti di Blender; dan tambalan datar per kelas untuk bantalan legrest (tanpa UV)
base.save('_tmp_tex/base2048.jpg',quality=90)
patches={}
Lg=0.299*A[...,0]+0.587*A[...,1]+0.114*A[...,2]
for c in range(3):
    for w in (28,22,16):
        okc=uniform_filter((C==c).astype(np.float32),w)>0.999
        okc[:64]=False; okc[-64:]=False; okc[:,:64]=False; okc[:,-64:]=False
        if not okc.any(): continue
        vL=uniform_filter(Lg**2,w)-uniform_filter(Lg,w)**2
        nvv=sum(uniform_filter(N[...,k]**2,w)-uniform_filter(N[...,k],w)**2 for k in range(3))
        sc=np.where(okc,vL/(vL[okc].mean()+1e-9)+nvv/(nvv[okc].mean()+1e-9),1e9)
        y,x=np.unravel_index(sc.argmin(),sc.shape); patches[c]={'x':int(x),'y':int(y),'w':w}; break
print('patches',patches)
json.dump({'H':H,'patches':patches},open('_tmp_tex/flatuv2.json','w'))
