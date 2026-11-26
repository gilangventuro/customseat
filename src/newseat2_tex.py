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
# benang: deret manik terang di alur jahit (4096 px). White top-hat -> manik, lalu dilebarkan jadi pita menyambung, buang bercak kecil.
from scipy.ndimage import grey_opening, binary_erosion, label
B4=np.asarray(Image.open('_tmp_tex/base.png').convert('RGB')).astype(np.float32); L4=0.299*B4[...,0]+0.587*B4[...,1]+0.114*B4[...,2]
beads=(L4-grey_opening(L4,size=(9,9)))>16
band=binary_erosion(binary_dilation(beads,iterations=3),iterations=2)
lab,n=label(band); cnt=np.bincount(lab.ravel()); keepc=cnt>=300; keepc[0]=False; band=keepc[lab]
# pita tebal -> garis tengahnya (skeleton) -> dilebarkan tipis, supaya benang halus seperti di model 3D
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes, convolve
band=binary_fill_holes(band)
band=gaussian_filter(band.astype(np.float32),3.0)>0.5          # haluskan tepi pita sebelum diambil garis tengahnya
sk=skeletonize(band)
K8=np.ones((3,3),int); K8[1,1]=0
for _ in range(int(os.environ.get('PRUNE','30'))):             # buang cabang pendek (duri) hasil skeleton
    nb=convolve(sk.astype(int),K8,mode='constant'); sk=sk&~((nb<=1))
thin=binary_dilation(sk,iterations=int(os.environ.get('THREAD_W','3')))
M2=np.asarray(Image.fromarray((thin*255).astype(np.uint8)).resize((H,H),Image.BOX)).astype(np.float32)/255
S=M2>0.3
LEVEL,THREAD=0.80,0.96
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
# peta benang (putih=benang) untuk warna jahitan yang bisa diganti di situs (shader di dummy-kursi-twinseat.html)
mask=np.clip(M2*1.8,0,1)
Image.fromarray((mask*255).astype(np.uint8)).save('_tmp_tex/thread-mask.webp',quality=90)
print('thread mask coverage',float((mask>0.5).mean()))
