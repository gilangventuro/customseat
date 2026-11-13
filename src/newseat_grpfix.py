# Buang potongan kecil (<50 vertex) yang nyasar dari vertex group 'Handrest Kanan' lalu simpan.
import bpy, numpy as np, sys
SRC=sys.argv[sys.argv.index('--')+1]; GN='Handrest Kanan'
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
me=o.data; nv=len(me.vertices)
ne=len(me.edges); e=np.empty(ne*2,np.int32); me.edges.foreach_get('vertices',e); e=e.reshape(-1,2)
g=o.vertex_groups[GN]; m=np.zeros(nv,bool)
for v in me.vertices:
    for ge in v.groups:
        if ge.group==g.index and ge.weight>0: m[v.index]=True
par=np.arange(nv)
def f(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for a,b in e:
    if m[a] and m[b]:
        ra,rb=f(a),f(b)
        if ra!=rb: par[ra]=rb
roots={}
for i in np.where(m)[0]: roots.setdefault(f(i),[]).append(int(i))
rm=[i for l in roots.values() if len(l)<50 for i in l]
g.lock_weight=False
g.remove(rm); print('dibuang',len(rm),'sisa',m.sum()-len(rm))
bpy.ops.wm.save_mainfile()
