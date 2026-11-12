# Tekstur diselaraskan dgn label face: panel jadi hitam seragam, tint-base per kelas memakai peta label per-texel.
from PIL import Image, ImageFilter
import numpy as np, sys
from scipy.ndimage import gaussian_filter
Image.MAX_IMAGE_PIXELS=None
ACC_R,ACC_G,ACC_B=[float(x) for x in (sys.argv[3].split(',') if len(sys.argv)>3 else '28,68,68'.split(','))]
d=sys.argv[1]; pre=sys.argv[2]
T=np.load('_tmp_texlabel.npy'); R=4096
base=Image.open(f'{d}/{pre}.png').convert('RGB').resize((R,R),Image.LANCZOS)
A=np.asarray(base).astype(np.float32); r,g,b=A[...,0],A[...,1],A[...,2]
L=0.299*r+0.587*g+0.114*b
# fallback texel tak terlabel: aturan kroma lama (cokelat -> pola)
Tf=T.copy()
chroma_pat=((r-b)>=6)&(r>g)
Tf=np.where(T==255,np.where(chroma_pat,1,0),T).astype(np.uint8)
out=A.copy()
pat=Tf==1; main=(Tf==0)|(Tf==4)   # kelas 4 (jahitan/Tali) warna dasarnya seperti kulit utama; bedanya hanya di tint
# panel: seragam hitam, detail jahitan dipertahankan (hp) -> tidak ada sisa hijau/cokelat
mean_p=L[pat].mean(); out_p=np.clip(31+(L-mean_p),6,120)
out[pat]=np.stack([out_p*1.03,out_p*0.99,out_p],-1)[pat]
# kulit utama: netralkan sisa rona hijau/cokelat
from scipy.ndimage import binary_dilation, uniform_filter
Lm=L.copy(); med=np.median(L[main])
bad=binary_dilation(main&(((g-r)>5)|((b-r)>5)|(L>np.percentile(L[main],97))),iterations=4)&main   # sisa toska/terang dekat list aksen
Lm[bad]=med; soft=gaussian_filter(bad.astype(np.float32),2.0); Lm=Lm*(1-0)+0   # ganti lalu haluskan tepi
Lm=np.where(bad,gaussian_filter(np.where(bad,med,L),1.5),Lm)
Lm=np.minimum(Lm,np.percentile(L[main],97))
gm=np.stack([Lm*1.03,Lm*0.99,Lm],-1); out[main]=gm[main]
# list aksen: toska merata (warna default pilihan user), detail relief tetap
acc=Tf==2
ACC=np.array([ACC_R,ACC_G,ACC_B],np.float32)
la=L[acc]; k=np.clip(L/ (np.percentile(la,50)+1e-6),0.5,1.6)
out[acc]=(ACC[None,:]*k[acc][:,None])
# ujung tuas putih -> merah (wajah dipilih di _tmp_whitetip.npy, warna merah diambil dari badan tuas)
import os
if os.path.exists('_tmp_whitetip.npy'):
    from PIL import ImageDraw
    UVL=np.load('_tmp_uvl.npy'); wf=np.load('_tmp_whitetip.npy'); rf=np.load('_tmp_red.npy')
    mk=Image.new('L',(R,R),0); dr=ImageDraw.Draw(mk)
    for f in wf: dr.polygon([(float(u%1*(R-1)),float((1-v)%1*(R-1))) for u,v in UVL[f]],fill=255)
    mk=np.asarray(mk.filter(ImageFilter.MaxFilter(3)))>0
    mr=Image.new('L',(R,R),0); dr=ImageDraw.Draw(mr)
    for f in np.where(rf)[0]: dr.polygon([(float(u%1*(R-1)),float((1-v)%1*(R-1))) for u,v in UVL[f]],fill=255)
    mr=np.asarray(mr)>0
    redc=np.median(A[mr],0); redL=np.median(L[mr])
    kk=np.clip(L/(np.median(L[mk])+1e-6),0.6,1.15)
    out[mk]=np.clip(redc[None,:]*kk[mk][:,None]*0.95,0,255)
    print('tip recolor',mk.sum(),redc)
Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save('_tmp_base4k.jpg',quality=88)
# tint-base: normalisasi per kelas berdasar peta label (2048)
H=2048; Ts=np.asarray(Image.fromarray(Tf).resize((H,H),Image.NEAREST))
Ls=np.asarray(Image.fromarray(L.astype(np.float32)).resize((H,H),Image.BOX)).astype(np.float64)
W=np.stack([np.asarray(Image.fromarray(((Ts==c)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))).astype(float)/255 for c in range(5)],0)
W/=W.sum(0,keepdims=True)+1e-6
AMP={0:0.045,1:0.06,2:0.09,3:0.09,4:0.06}; tint=np.zeros((H,H))
for c in range(5):
    sel=Ts==c
    if sel.sum()<10: continue
    m=sel.astype(np.float64); Lm=np.where(sel,Ls,0.0)
    lowp=gaussian_filter(Lm,10)/(gaussian_filter(m,10)+1e-6)   # blur ber-mask: tidak bocor lewat tepi pulau UV
    hp=np.where(sel,Ls-lowp,0.0); hp=np.clip(hp/(hp[sel].std()+1e-6),-2,2)
    tint+=W[c]*np.clip(0.88+AMP[c]*hp,0.55,1.0)
Image.fromarray(np.clip(tint*255,0,255).astype(np.uint8)).resize((R,R),Image.BICUBIC).convert('RGB').save('_tmp_tint4k.webp',quality=82)
print('ok')
