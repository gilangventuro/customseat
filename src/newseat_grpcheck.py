# Analisis komponen terhubung tiap vertex group di kursi-captain-pisah.blend (read-only).
import bpy, numpy as np, sys
SRC=sys.argv[sys.argv.index('--')+1]
bpy.ops.wm.open_mainfile(filepath=SRC)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
bpy.context.view_layer.objects.active=o
if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
me=o.data; nv=len(me.vertices)
ne=len(me.edges); e=np.empty(ne*2,np.int32); me.edges.foreach_get('vertices',e); e=e.reshape(-1,2)
co=np.empty(nv*3); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
for g in o.vertex_groups:
    m=np.zeros(nv,bool)
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
    for i in np.where(m)[0]: roots.setdefault(f(i),[]).append(i)
    print('GRUP',g.name,'verts',m.sum(),'komponen',len(roots))
    for r,l in sorted(roots.items(),key=lambda x:-len(x[1]))[:8]:
        c=co[l].mean(0); print('   n=%d centroid=(%.3f,%.3f,%.3f)'%(len(l),*c))
