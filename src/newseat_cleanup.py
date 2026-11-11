# Bersihkan label hasil edit manual: pulau kecil -> digabung ke tetangga dominan, gigi 1-face dihaluskan.
import numpy as np, sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def face_adj(pv):
    n=len(pv); e=np.concatenate([pv[:,[0,1]],pv[:,[1,2]],pv[:,[2,0]]]); fid=np.tile(np.arange(n),3)
    e=np.sort(e,1); key=e[:,0].astype(np.int64)*(e.max()+1)+e[:,1]
    o=np.argsort(key,kind='stable'); key=key[o]; fid=fid[o]
    same=key[1:]==key[:-1]
    a,b=fid[:-1][same],fid[1:][same]
    return a,b
def cleanup(lab,pv,min_size=200,verbose=True):
    lab=lab.copy(); a,b=face_adj(pv); n=len(lab)
    for it in range(3):
        # 1) pulau kecil
        m=lab[a]==lab[b]
        G=coo_matrix((np.ones(m.sum()),(a[m],b[m])),shape=(n,n))
        nc,cl=connected_components(G,directed=False)
        sz=np.bincount(cl); small=np.where(sz<min_size)[0]
        ch=0
        if len(small):
            isS=np.zeros(nc,bool); isS[small]=True
            cross=~m
            ea,eb=a[cross],b[cross]
            for src,dst in ((ea,eb),(eb,ea)):
                pass
            # suara tetangga per komponen kecil
            votes={}
            for s,t in ((ea,eb),(eb,ea)):
                sel=isS[cl[s]]
                for c_,l_ in zip(cl[s[sel]],lab[t[sel]]):
                    votes.setdefault(c_,np.zeros(4,int))[l_]+=1
            new=np.zeros(nc,np.int8)-1
            for c_,v in votes.items(): new[c_]=v.argmax()
            fix=isS[cl]&(new[cl]>=0)
            ch=(lab[fix]!=new[cl][fix]).sum(); lab[fix]=new[cl][fix]
        # 2) gigi 1-face: >=2 dari 3 tetangga sama & beda dari diri
        cnt=np.zeros((n,4),int)
        for s,t in ((a,b),(b,a)): np.add.at(cnt,(s,lab[t]),1)
        best=cnt.argmax(1); bc=cnt.max(1)
        tooth=(bc>=2)&(best!=lab)&(cnt[np.arange(n),lab]<=1)
        lab[tooth]=best[tooth]
        if verbose: print('iter',it,'pulau diubah',ch,'gigi',tooth.sum())
    return lab
if __name__=='__main__':
    d=np.load('_tmp_new.npz'); pv=d['pv']; lab=np.load(sys.argv[1]).astype(np.int8)
    out=cleanup(lab,pv,int(sys.argv[3]) if len(sys.argv)>3 else 200)
    print('berubah',(out!=lab).sum(),np.bincount(out)); np.save(sys.argv[2],out)
