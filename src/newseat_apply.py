# Langkah D (Blender): buka hasil decimate, pasang 4 material per-label + tekstur, ekspor GLB Draco
import bpy, numpy as np, os, sys
a=sys.argv[sys.argv.index('--')+1:]
BLEND,LAB,BASE,NORM,OUT=a
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
lab=np.load(LAB).astype(np.int32)
# Tuas merah di celah kursi: duplikat persis (geser x +DX, tanpa cermin) ke kursi sebelahnya bila LEVER_NPY diberikan
RED=os.environ.get('LEVER_NPY')
if RED:
    import bmesh
    DX=float(os.environ.get('LEVER_DX','0.827')); sel=np.load(RED); me0=o.data
    bm=bmesh.new(); bm.from_mesh(me0); bm.faces.ensure_lookup_table()
    uvl=bm.loops.layers.uv.active
    vmap={}
    def mv(v):
        if v.index not in vmap:
            vmap[v.index]=bm.verts.new((v.co.x+DX,v.co.y,v.co.z))
        return vmap[v.index]
    src=[bm.faces[i] for i in sel]
    newf=[]
    for f in src:
        vs=[mv(v) for v in f.verts]
        nf=bm.faces.new(vs)
        for l_new,l_old in zip(nf.loops, list(f.loops)):
            l_new[uvl].uv=l_old[uvl].uv
        newf.append(nf)
    bm.to_mesh(me0); bm.free(); me0.update()
    print('lever faces mirrored',len(sel))
    lab=np.concatenate([lab,lab[sel]])
me=o.data; me.materials.clear()
base=bpy.data.images.load(BASE); norm=bpy.data.images.load(NORM); norm.colorspace_settings.name='Non-Color'
def mk(name,rough):
    m=bpy.data.materials.new(name); m.use_nodes=True
    nt=m.node_tree; nt.nodes.clear()
    out=nt.nodes.new('ShaderNodeOutputMaterial'); bs=nt.nodes.new('ShaderNodeBsdfPrincipled')
    bs.inputs['Roughness'].default_value=rough; bs.inputs['Metallic'].default_value=0.0
    tb=nt.nodes.new('ShaderNodeTexImage'); tb.image=base
    tn=nt.nodes.new('ShaderNodeTexImage'); tn.image=norm
    nm=nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value=1.0
    nt.links.new(tb.outputs['Color'],bs.inputs['Base Color'])
    nt.links.new(tn.outputs['Color'],nm.inputs['Color']); nt.links.new(nm.outputs['Normal'],bs.inputs['Normal'])
    nt.links.new(bs.outputs['BSDF'],out.inputs['Surface'])
    return m
for name,r in [('leather_main',0.55),('leather_pattern',0.55),('leather_accent',0.55),('leather_rest',0.6)]:
    me.materials.append(mk(name,r))
lab=np.where(lab==4,0,lab)   # jahitan = geometri benang terpisah (THREAD_NPZ), bukan label material
me.polygons.foreach_set('material_index',lab); me.update()
PN=os.environ.get('PART_NPY'); NAMES={1:'Handrest_Kanan',2:'Handrest_Kiri',3:'Footrest_Kanan',4:'Footrest_Kiri'}
part=np.zeros(len(me.polygons),np.int8)
if PN:
    pp=np.load(PN); m_=min(len(pp),len(part)); part[:m_]=pp[:m_]
TH=os.environ.get('THREAD_NPZ')
sm=None
if TH:
    sm=bpy.data.materials.new('leather_stitch'); sm.use_nodes=True
    b=sm.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value=(0.03,0.032,0.036,1); b.inputs['Roughness'].default_value=0.55
    z=np.load(TH); V=z['V']; F=z['F']
    qpart=np.load(os.environ['QPART_NPY']) if os.environ.get('QPART_NPY') else np.zeros(len(V)//4,np.int8)   # bagian per quad (dihitung di luar Blender: src/newseat_qpart.py)
    fpart=qpart[np.arange(len(F))//2]     # 2 segitiga per quad
    for k in sorted(set(fpart.tolist())):
        sel=F[fpart==k]; used=np.unique(sel); remap=-np.ones(len(V),np.int64); remap[used]=np.arange(len(used))
        tm=bpy.data.meshes.new('JahitanMesh%d'%k); tm.from_pydata(V[used].tolist(),[],remap[sel].tolist()); tm.update()
        to=bpy.data.objects.new('Jahitan' if k==0 else 'Jahitan_'+NAMES[k],tm); bpy.context.scene.collection.objects.link(to)
        tm.materials.append(sm)
        for pl in tm.polygons: pl.use_smooth=True
if PN:
    import bmesh
    for k,nm in NAMES.items():
        bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=k],context='FACES')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        nm_=bpy.data.meshes.new(nm); bm.to_mesh(nm_); bm.free()
        for m in me.materials: nm_.materials.append(m)
        ob=bpy.data.objects.new(nm,nm_); bpy.context.scene.collection.objects.link(ob)
        print('bagian',nm,len(nm_.polygons))
    bm=bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm,geom=[f for f in bm.faces if part[f.index]!=0],context='FACES')
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
    bm.to_mesh(me); bm.free()
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
