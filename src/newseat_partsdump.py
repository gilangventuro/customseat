# Baca pembagian komponen (vertex group) dari kursi-captain-pisah.blend -> id bagian per-face (0=badan).
import bpy, numpy as np, sys
SRC,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')   # file disimpan saat Edit Mode
me=o.data; nv=len(me.vertices); nf=len(me.polygons)
MAP={'handrest kanan':1,'handrest kiri':2,'footrest_kanan':3,'footrest_kiri':4,'handrest tengah':5}
mem=np.zeros((6,nv),bool)
for g in o.vertex_groups:
    k=MAP.get(g.name.strip().lower())
    if k is None: print('abaikan grup',g.name); continue
    for v in me.vertices:
        for ge in v.groups:
            if ge.group==g.index and ge.weight>0: mem[k,v.index]=True
# buang potongan kecil (<50 titik) yang nyasar dari tiap grup
ne=len(me.edges); ed=np.empty(ne*2,np.int32); me.edges.foreach_get('vertices',ed); ed=ed.reshape(-1,2)
for k in range(1,6):
    m=mem[k]; par=np.arange(nv)
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for a,b in ed[m[ed[:,0]]&m[ed[:,1]]]:
        ra,rb=f(a),f(b)
        if ra!=rb: par[ra]=rb
    roots={}
    for i in np.where(m)[0]: roots.setdefault(f(i),[]).append(i)
    for l in roots.values():
        if len(l)<50: mem[k,l]=False; print('buang nyasar',k,len(l))
pv=np.empty(nf*3,np.int32); me.polygons.foreach_get('vertices',pv); pv=pv.reshape(nf,3)
cnt=np.stack([mem[k][pv].sum(1) for k in range(6)],1); cnt[:,0]=0
part=np.where(cnt.max(1)>=2,cnt.argmax(1),0).astype(np.int8)
np.save(OUT,part); print('part',np.bincount(part,minlength=6))
