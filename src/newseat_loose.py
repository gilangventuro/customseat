# Daftar komponen terhubung kecil (serpihan) pada seluruh mesh, beserta part-nya.
import bpy, numpy as np, sys
SRC=sys.argv[sys.argv.index('--')+1]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
me=o.data; nv=len(me.vertices); nf=len(me.polygons)
pv=np.empty(nf*3,np.int32); me.polygons.foreach_get('vertices',pv); pv=pv.reshape(nf,3)
co=np.empty(nv*3); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
par=np.arange(nv)
def f(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for a,b,c in pv:
    for x,y in ((a,b),(b,c)):
        rx,ry=f(x),f(y)
        if rx!=ry: par[rx]=ry
roots=np.array([f(i) for i in range(nv)])
u,cnt=np.unique(roots,return_counts=True)
for r,c in sorted(zip(u,cnt),key=lambda t:t[1]):
    if c<3000:
        m=roots==r; print('komp verts=%d centroid=(%.3f,%.3f,%.3f)'%(c,*co[m].mean(0)))
