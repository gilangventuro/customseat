# Jahitan baru: ambil jalur dari deretan manik di foto scan (4096), rangkai jadi garis tipis (skeleton), buang goresan pendek,
# lalu gambar ulang manik seragam sepanjang garis. Keluaran: _tmp_tex/stitch2048.npy (float 0..1, 2048x2048) + _tmp_tex/stitch_prev.png
from PIL import Image
import numpy as np, os
from scipy.ndimage import grey_opening, label, find_objects, gaussian_filter, binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize
Image.MAX_IMAGE_PIXELS=None
B=np.asarray(Image.open('_tmp_tex/base.png').convert('RGB')).astype(np.float32); L=0.299*B[...,0]+0.587*B[...,1]+0.114*B[...,2]
H=L.shape[0]
T=float(os.environ.get('BEAD_T','28')); SP=float(os.environ.get('SPACING','11')); R=float(os.environ.get('RAD','3.2')); MINLEN=int(os.environ.get('MINLEN','90')); GAP=float(os.environ.get('LINK','5'))
raw=(L-grey_opening(L,size=(9,9)))>T
lab,n=label(raw); cnt=np.bincount(lab.ravel()); keep=(cnt>=6)&(cnt<=140); keep[0]=False
for i,sl in enumerate(find_objects(lab),1):
    if not keep[i]: continue
    hh=sl[0].stop-sl[0].start; ww=sl[1].stop-sl[1].start
    if max(hh,ww)>16 or max(hh,ww)>3.2*max(1,min(hh,ww)): keep[i]=False
beads=keep[lab]; print('beads',int(keep.sum()))
# sambung manik jadi garis: blur lalu ambang -> skeleton
line=gaussian_filter(beads.astype(np.float32),GAP)>float(os.environ.get('LT','0.045'))
sk=skeletonize(line)
# buang komponen skeleton pendek (bercak/goresan)
lb,m=label(sk,structure=np.ones((3,3))); c2=np.bincount(lb.ravel()); ok=c2>=MINLEN; ok[0]=False; sk=ok[lb]; print('skeleton px',int(sk.sum()),'komponen',int(ok.sum()))
# manik seragam sepanjang skeleton: jarak geodesik dari satu ujung per komponen, titik tiap SP piksel
ys,xs=np.nonzero(sk); idx=-np.ones(sk.shape,np.int64); idx[ys,xs]=np.arange(len(ys))
rows=[];cols=[];wts=[]
for dy,dx,w in ((0,1,1),(1,0,1),(1,1,1.4142),(1,-1,1.4142)):
    y2=ys+dy; x2=xs+dx; v=(y2<H)&(x2>=0)&(x2<H); j=idx[y2[v],x2[v]]; k=j>=0
    rows+=list(np.nonzero(v)[0][k]); cols+=list(j[k]); wts+=[w]*int(k.sum())
G=coo_matrix((wts,(rows,cols)),shape=(len(ys),len(ys))).tocsr(); G=G+G.T
lbl=lb[ys,xs]; dots=[]
for comp in np.unique(lbl):
    nodes=np.nonzero(lbl==comp)[0]
    # ujung: node berderajat 1 (jika ada), kalau tidak sembarang
    deg=np.diff(G.tocsr().indptr)[nodes]; ends=nodes[deg==1]; root=ends[0] if len(ends) else nodes[0]
    d,pred=dijkstra(G,directed=False,indices=root,return_predecessors=True)
    dn=d[nodes]; pn=pred[nodes]
    for a,(nd,p) in zip(nodes,zip(dn,pn)):
        if not np.isfinite(nd): continue
        pd_=d[p] if p>=0 else -1
        if p<0 or int(nd//SP)>int(pd_//SP): dots.append(a)
dots=np.array(dots); print('dots',len(dots))
S=np.zeros((H,H),np.float32); S[ys[dots],xs[dots]]=1
S=gaussian_filter(S,R*0.6); S=np.clip(S/S.max()*float(os.environ.get('GAIN','3.0')),0,1)
S2=np.asarray(Image.fromarray((S*255).astype(np.uint8)).resize((2048,2048),Image.BOX)).astype(np.float32)/255
np.save('_tmp_tex/stitch2048.npy',S2)
prev=np.asarray(Image.open('_tmp_tex/base2048.jpg')).copy(); m=S2>0.15; prev[m]=(245,235,200)
Image.fromarray(prev).save('_tmp_tex/stitch_prev.png'); print('selesai')
