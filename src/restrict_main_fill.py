import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def fill_panels(c, ns, lab, RES=0.0008, CLOSE=14, DTOL=0.009, verbose=True):
    x,y,z=c[:,0],c[:,1],c[:,2]
    side=(x>0)
    part=np.full(len(c),-1)
    part[(ns[:,1]<-0.55)&(y>-0.021)&(z>0.052)]=0   # sandaran depan
    part[(ns[:,2]>0.55)&(y>-0.021)&(z<0.056)]=1    # dudukan atas
    part[(ns[:,2]>0.3)&(y<=-0.021)]=2              # footrest
    out=lab.copy()
    main_idx=np.where(lab==0)[0]
    for sd in (False,True):
        for pt in (0,1,2):
            G=(side==sd)&(part==pt)
            pat=np.where(G&((lab==1)|(lab==2)))[0]
            if len(pat)<2000: print('skip',sd,pt,len(pat)); continue
            pts=c[pat]; cen=pts.mean(0)
            u,s_,vt=np.linalg.svd(pts-cen,full_matrices=False)
            e1,e2,pn=vt[0],vt[1],vt[2]
            pnm=ns[pat].mean(0); pnm/=np.linalg.norm(pnm)
            if pn@pnm<0: pn=-pn
            a=(pts-cen)@e1; b=(pts-cen)@e2
            amin,bmin=a.min()-0.01,b.min()-0.01
            W=int((a.max()+0.01-amin)/RES)+2;H=int((b.max()+0.01-bmin)/RES)+2
            m=np.zeros((H,W),bool); m[((b-bmin)/RES).astype(int),((a-amin)/RES).astype(int)]=True
            m=ndimage.binary_closing(m,structure=np.ones((CLOSE,CLOSE)))
            m=ndimage.binary_fill_holes(m)
            mi=main_idx[G[main_idx]]
            d=(c[mi]-cen)
            aa=((d@e1-amin)/RES).astype(int); bb=((d@e2-bmin)/RES).astype(int)
            ok=(aa>=0)&(aa<W)&(bb>=0)&(bb<H)
            sel=np.zeros(len(mi),bool); sel[ok]=m[bb[ok],aa[ok]]
            sel&=(np.abs(d@pn)<DTOL)
            out[mi[sel]]=1
            if verbose: print('panel',sd,pt,len(pat),'flipped',sel.sum(),'of',len(mi))
    return out
