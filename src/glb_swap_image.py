# Ganti isi gambar bernama <nama> di dalam GLB dengan file JPEG/PNG baru (tanpa ekspor ulang dari Blender).
import json,struct,sys
src,dst,name,fn=sys.argv[1:5]
b=open(src,'rb').read(); jl=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+jl]); p=20+jl
bl=struct.unpack('<I',b[p:p+4])[0]; bin0=b[p+8:p+8+bl]
im=[i for i in j['images'] if i.get('name')==name][0]; newb=open(fn,'rb').read()
out=bytearray(); order=sorted(range(len(j['bufferViews'])),key=lambda i:j['bufferViews'][i].get('byteOffset',0))
for i in order:
    v=j['bufferViews'][i]; d=newb if i==im['bufferView'] else bin0[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
    while len(out)%4: out+=b'\0'
    v['byteOffset']=len(out); v['byteLength']=len(d); out+=d
while len(out)%4: out+=b'\0'
j['buffers'][0]['byteLength']=len(out)
js=json.dumps(j,separators=(',',':')).encode()
while len(js)%4: js+=b' '
open(dst,'wb').write(struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(out))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(out),0x004E4942)+bytes(out))
print('ok',len(out)/1e6)
