# Perkecil tekstur kayu (satinfinishwalnut*) di dalam GLB tanpa mengekspor ulang: 1024 px q99 -> 512 px q85.
import json,struct,io,sys
from PIL import Image
src,dst=sys.argv[1],sys.argv[2]; SIZE=int(sys.argv[3]) if len(sys.argv)>3 else 512
b=open(src,'rb').read(); jl=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+jl]); p=20+jl
bl=struct.unpack('<I',b[p:p+4])[0]; bin0=b[p+8:p+8+bl]
wood={t['source'] for m in j['materials'] if 'walnut' in m['name'] for t in [j['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']]]}
newbytes={}
for si in wood:
    v=j['bufferViews'][j['images'][si]['bufferView']]; d=bin0[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
    im=Image.open(io.BytesIO(d)).convert('RGB').resize((SIZE,SIZE),Image.LANCZOS); o=io.BytesIO(); im.save(o,'JPEG',quality=85,optimize=True)
    newbytes[j['images'][si]['bufferView']]=o.getvalue(); print('kayu',si,len(d)//1000,'->',len(o.getvalue())//1000,'KB')
out=bytearray(); order=sorted(range(len(j['bufferViews'])),key=lambda i:j['bufferViews'][i].get('byteOffset',0))
for i in order:
    v=j['bufferViews'][i]; d=newbytes.get(i) or bin0[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
    while len(out)%4: out+=b'\0'
    v['byteOffset']=len(out); v['byteLength']=len(d); out+=d
while len(out)%4: out+=b'\0'
j['buffers'][0]['byteLength']=len(out)
js=json.dumps(j,separators=(',',':')).encode()
while len(js)%4: js+=b' '
res=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(out))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(out),0x004E4942)+bytes(out)
open(dst,'wb').write(res); print('GLB',len(b)/1e6,'->',len(res)/1e6,'MB')
