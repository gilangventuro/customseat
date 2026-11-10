# Langkah 0: bikin peta kelas dari tekstur asli (0 hitam,1 cokelat,2 hijau,3 merah) pakai selisih kroma (tahan bayangan gelap)
from PIL import Image, ImageFilter
import numpy as np, sys
Image.MAX_IMAGE_PIXELS=None
src=sys.argv[1]; R=2048
im=Image.open(src).convert('RGB').resize((R,R),Image.BOX).filter(ImageFilter.GaussianBlur(1.2))
a=np.asarray(im).astype(float); r,g,b=a[...,0],a[...,1],a[...,2]
lab=np.zeros((R,R),np.uint8)
lab[((r-b)>=7)&((r-g)>=3)&(r>g)]=1
lab[((g-r)>=5)&((b-r)>=3)&((g-b)<=14)]=2
lab[((r-g)>45)&((r-b)>35)]=3
lab=np.asarray(Image.fromarray(lab).filter(ImageFilter.ModeFilter(5)))
np.save('_tmp_labmap.npy',lab); print([round((lab==i).mean(),3) for i in range(4)])
