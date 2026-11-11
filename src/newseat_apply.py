# Langkah D (Blender): buka hasil decimate, pasang 4 material per-label + tekstur, ekspor GLB Draco
import bpy, numpy as np, os, sys
a=sys.argv[sys.argv.index('--')+1:]
BLEND,LAB,BASE,NORM,OUT=a
bpy.ops.wm.open_mainfile(filepath=BLEND)
o=[x for x in bpy.data.objects if x.type=='MESH'][0]
lab=np.load(LAB).astype(np.int32)
# Tuas merah hanya ada di sisi kanan: salin cermin (x -> -x) ke kursi kiri bila ada file wajah merah
RED=os.environ.get('LEVER_NPY')
if RED:
    import bmesh
    sel=np.load(RED); me0=o.data
    bm=bmesh.new(); bm.from_mesh(me0); bm.faces.ensure_lookup_table()
    uvl=bm.loops.layers.uv.active
    vmap={}
    def mv(v):
        if v.index not in vmap:
            vmap[v.index]=bm.verts.new((-v.co.x,v.co.y,v.co.z))
        return vmap[v.index]
    src=[bm.faces[i] for i in sel]
    newf=[]
    for f in src:
        vs=[mv(v) for v in f.verts]
        nf=bm.faces.new(vs[::-1])
        for l_new,l_old in zip(nf.loops, [f.loops[i] for i in range(len(f.loops))][::-1]):
            l_new[uvl].uv=l_old[uvl].uv
        newf.append(nf)
    bm.to_mesh(me0); bm.free(); me0.update()
    print('lever faces mirrored',len(sel))
    lab=np.concatenate([lab,np.full(len(sel),3,np.int32)])
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
me.polygons.foreach_set('material_index',lab); me.update()
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
