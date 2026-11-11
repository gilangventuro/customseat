# .blend untuk memisahkan armrest & footrest secara manual: material warna-warni + vertex group kandidat.
import bpy, bmesh, numpy as np, sys, os
BLEND,LAB,BASE,LEVER,OUT=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]; o.name='Kursi'; me=o.data
lab=np.load(LAB).astype(np.int32); sel=np.load(LEVER)
cen=np.array([p.center[:] for p in me.polygons])
# salinan tuas (sama seperti di situs)
bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table(); uvl=bm.loops.layers.uv.active; vm={}
def mv(v):
    if v.index not in vm: vm[v.index]=bm.verts.new((v.co.x+0.827,v.co.y,v.co.z))
    return vm[v.index]
srcf=[bm.faces[int(i)] for i in sel]
for f in srcf:
    nf=bm.faces.new([mv(v) for v in f.verts])
    for a,b in zip(nf.loops,f.loops): a[uvl].uv=b[uvl].uv
bm.to_mesh(me); bm.free(); me.update()
lab=np.concatenate([lab,lab[sel]]); cen=np.concatenate([cen,cen[sel]+np.array([0.827,0,0])])
me.materials.clear(); img=bpy.data.images.load(BASE); img.pack()
cols={'1_KulitUtama_hitam':(0.12,0.12,0.12),'2_MotifBerlian_panel':(0.9,0.5,0.1),'3_ListAksen_hijau':(0.1,0.8,0.7),'4_TidakBisaDiganti':(0.9,0.1,0.1)}
for n,c in cols.items():
    m=bpy.data.materials.new(n); m.use_nodes=True; m.diffuse_color=(*c,1)
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*c,1); me.materials.append(m)
me.polygons.foreach_set('material_index',lab); me.update()
# vertex group kandidat (perkiraan awal -- user merapikan lalu Separate)
cand={'footrest_kiri':(cen[:,1]<-0.5)&(cen[:,0]<0),
      'footrest_kanan':(cen[:,1]<-0.5)&(cen[:,0]>=0),
      'armrest_kiri_PERKIRAAN':(lab==3)&(cen[:,0]<-0.62)&(cen[:,1]>-0.3)&(cen[:,2]>-0.35),
      'armrest_kanan_PERKIRAAN':(lab==3)&(cen[:,0]>0.62)&(cen[:,1]>-0.3)&(cen[:,2]>-0.35)}
for n,m in cand.items():
    vg=o.vertex_groups.new(name=n); vs=set()
    for p in np.where(m)[0]: vs.update(me.polygons[int(p)].vertices)
    vg.add(list(vs),1.0,'REPLACE'); print(n,m.sum())
for scr in bpy.data.screens:
    for a in scr.areas:
        if a.type=='VIEW_3D':
            for sp in a.spaces:
                if sp.type=='VIEW_3D': sp.shading.type='SOLID'; sp.shading.color_type='MATERIAL'
bpy.ops.wm.save_as_mainfile(filepath=OUT,compress=True); print('done',os.path.getsize(OUT)/1e6)
