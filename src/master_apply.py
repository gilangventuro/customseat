# Ekspor model twin seat dari "SEATS MASTER ALLDILA.blend" (master buatan pengguna) ke GLB untuk situs.
# Pakai:  blender --background --python src/master_apply.py -- "SEATS MASTER ALLDILA.blend" OUT.glb
# Kontrak dengan dummy-kursi-twinseat.html:
#   material kulit  : leather_main / leather_pattern / leather_accent / leather_stitch (warna = tint dari situs)
#   nama mesh       : *Handrest_Kiri|Kanan|Tengah*, *Footrest_Kiri|Kanan* (animasi lepas/pasang)
# Material master: 'Body Utama' -> main, 'Boddy Dalam' -> pattern, 'defaultplastic.001' -> accent.
import bpy,bmesh,mathutils,numpy as np,math,os,sys
a=sys.argv[sys.argv.index('--')+1:]; SRC,OUT=a[0],a[1]
env=lambda k,d: os.environ.get(k,d)
bpy.ops.wm.open_mainfile(filepath=SRC)
for _o in list(bpy.data.objects):   # blend bisa tersimpan saat objek masih di Edit Mode -> keluar dulu sebelum mesh diubah
    if _o.type=='MESH' and _o.mode!='OBJECT':
        bpy.context.view_layer.objects.active=_o; bpy.ops.object.mode_set(mode='OBJECT')

BODY_T=int(env('BODY_TARGET','520000'));  PEDAL_BASE_T=int(env('PEDAL_BASE_TARGET','160000')); FOOT_T=int(env('FOOT_TARGET','90000'))
HAND_R=float(env('HAND_RATIO','0.55'));   SHELL_R=float(env('SHELL_RATIO','0.7'));            CUP_R=float(env('CUP_RATIO','0.25'))
YCUT=float(env('FOOT_YCUT','-1.75'))      # bagian alas kaki: titik di depan garis ini (y Blender, depan = -y)

# ---- 1) kumpulkan, catat induk, bake transformasi dunia ke mesh ----
objs=[o for o in bpy.data.objects if o.type=='MESH']
_b0=next(o for o in objs if o.name.startswith('Meshy_AI_Twin_Luxe_Seats'))   # satu badan saja; salinan kerja (.001/.002) & bola uji jahitan di file master tidak ikut diekspor
_keep=[o for o in objs if o is _b0 or not (o.name.startswith('Meshy_AI_Twin_Luxe_Seats') or o.name.startswith('Icosphere'))]
for _o in [o for o in objs if o not in _keep]: bpy.data.objects.remove(_o)
objs=_keep
par={o.name:(o.parent.name if o.parent else None) for o in objs}
for o in objs:
    M=o.matrix_world.copy()
    if o.data.users>1: o.data=o.data.copy()
    if o.name.startswith('rev.3-HANDLE'): sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import master_handlecap; master_handlecap.cap(o)   # lubang soket baut di ujung ganggang ditutup rata (koordinat lokal)
    o.parent=None; o.matrix_world=mathutils.Matrix.Identity(4)
    o.data.transform(M)
    if M.determinant()<0: o.data.flip_normals()
for o in [x for x in bpy.data.objects if x.type=='EMPTY']: bpy.data.objects.remove(o)
print('mesh',len(objs),'face total',sum(len(o.data.polygons) for o in objs))

def bbox(o):
    n=len(o.data.vertices); P=np.empty(n*3); o.data.vertices.foreach_get('co',P); P=P.reshape(-1,3); return P.min(0),P.max(0)
body=next(o for o in objs if o.name.startswith('Meshy_AI_Twin_Luxe_Seats'))
bm_=bmesh.new(); bm_.from_mesh(body.data)   # buang sisa geometri liar di dekat lantai (z<0.5) yang kadang tertinggal di file master & menggeser titik tengah badan
st_=[v for v in bm_.verts if v.co.z<0.5]
if st_: print('buang geometri liar',len(st_)); bmesh.ops.delete(bm_,geom=st_,context='VERTS'); bm_.to_mesh(body.data); body.data.update()
bm_.free()
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import master_seamfix; master_seamfix.despike(body); master_seamfix.clean_thread(body); master_seamfix.repair(body)   # alur jahitan yang putus di blok kulit kanan-depan dibangun ulang
b0,b1=bbox(body); XC=float((b0[0]+b1[0])/2); print('badan x',b0[0],b1[0],'XC',XC)

# ---- 2) material ----
def mk(name,col,met,rough):
    m=bpy.data.materials.get(name)
    if m: return m
    m=bpy.data.materials.new(name); m.use_nodes=True; nt=m.node_tree; nt.nodes.clear()
    out=nt.nodes.new('ShaderNodeOutputMaterial'); bs=nt.nodes.new('ShaderNodeBsdfPrincipled')
    bs.inputs['Base Color'].default_value=(col[0],col[1],col[2],1); bs.inputs['Metallic'].default_value=met; bs.inputs['Roughness'].default_value=rough
    nt.links.new(bs.outputs['BSDF'],out.inputs['Surface']); m.diffuse_color=(col[0],col[1],col[2],1); return m
LEATHER={'Body Utama':'leather_main','Boddy Dalam':'leather_pattern','List Aksen':'leather_accent','defaultplastic.001':'leather_stitch','Jahitan':'leather_stitch',
         'Kulit Utama':'leather_main','Kulit Utama.001':'leather_main','kulit dalam':'leather_pattern','kulit dalam.001':'leather_pattern','kulit dalam.002':'leather_pattern',
         'Kulit Luar':'leather_main','Kulit Luar.001':'leather_main','list':'leather_accent','list.001':'leather_accent'}   # badan + bantalan footrest
MATDEF={'leather_main':((0.045,0.046,0.05),0.0,0.55),'leather_pattern':((0.045,0.046,0.05),0.0,0.55),'leather_accent':((0.02,0.40,0.075),0.0,0.5),'leather_stitch':((0.03,0.032,0.036),0.0,0.55),
        'plastik_hitam':((0.025,0.025,0.028),0.0,0.5),'plastik_abu':((0.2,0.2,0.2),0.0,0.5),'pad_hitam':((0.045,0.045,0.047),0.0,0.6),
        'part_handle':((0.025,0.025,0.028),0.0,0.45),'part_cupholder':((0.025,0.025,0.028),0.0,0.5),
        'baja':((0.5,0.5,0.5),1.0,0.25),'aluminium':((0.55,0.56,0.58),1.0,0.4),'tuas_merah':((0.6,0.03,0.02),0.0,0.45)}
WOOD={'kayu_atas':'satinfinishwalnut','kayu_samping':'satinfinishwalnut.001'}   # material kayu asli (tekstur Image_0 / Image_1) yang dipakai ulang
def cat(m,onbody):
    n=m.name if m else ''
    if onbody and n in LEATHER: return LEATHER[n]
    if n in ('Bahan.001','tuas'): return 'tuas_merah'
    if n.startswith('polished') or n in ('Material_5','Material_6'): return 'baja'
    if n.startswith('satinfinishalu'): return 'aluminium'
    if n.startswith('satinfinishwalnut'): return 'kayu_samping' if n.rsplit('.',1)[-1] in ('001','002','005') else 'kayu_atas'   # kayu bertekstur di armrest: dipertahankan
    bc=(0,0,0)
    if m and m.use_nodes:
        b=next((x for x in m.node_tree.nodes if x.type=='BSDF_PRINCIPLED'),None)
        if b: bc=tuple(b.inputs['Base Color'].default_value)[:3]
    sat=max(bc)-min(bc)
    if sat<0.12 and 0.12<max(bc)<0.5: return 'plastik_abu'      # abu-abu yang memang diatur pengguna
    return 'plastik_hitam'                                     # hitam / warna penanda -> hitam
def remap_mats(o,onbody=False,force=None):
    if force:   # bagian yang bisa diganti warna di situs: satu material khusus
        me=o.data; me.materials.clear(); me.materials.append(mk(force,*MATDEF[force])); me.polygons.foreach_set('material_index',np.zeros(len(me.polygons),np.int32)); me.update(); return
    me=o.data; old=list(me.materials) or [None]; cats=[cat(m,onbody) for m in old]
    mi=np.empty(len(me.polygons),np.int32); me.polygons.foreach_get('material_index',mi); mi=np.clip(mi,0,len(old)-1)
    if onbody:
        order=['leather_main','leather_pattern','leather_accent','leather_stitch']
        used=[c for c in order if any(cats[i]==c and (mi==i).any() for i in range(len(old)))]
    else: used=list(dict.fromkeys(cats))
    new=np.array([used.index(c) if c in used else 0 for c in cats],np.int32)
    me.materials.clear()
    for c in used: me.materials.append(bpy.data.materials[WOOD[c]] if c in WOOD else mk(c,*MATDEF[c]))
    me.polygons.foreach_set('material_index',new[mi]); me.update()

# ---- 3) pecah alas kaki (satu mesh Meshy: bagian depan = footrest, sisanya rangka dasar) ----
ped=next(o for o in objs if o.name.startswith('Meshy_AI_Dual_Foot_Pedal'))
me=ped.data; nf=len(me.polygons); C=np.empty(nf*3); me.polygons.foreach_get('center',C); C=C.reshape(-1,3)
masks={'Footrest_Kiri':(C[:,1]<YCUT)&(C[:,0]<XC),'Footrest_Kanan':(C[:,1]<YCUT)&(C[:,0]>=XC)}
masks['Rangka_Dasar']=~(masks['Footrest_Kiri']|masks['Footrest_Kanan'])
def subset(src,mask,name):
    bm=bmesh.new(); bm.from_mesh(src.data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm,geom=[f for f,k in zip(bm.faces,mask) if not k],context='FACES')
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
    nm=bpy.data.meshes.new(name); bm.to_mesh(nm); bm.free()
    for m in src.data.materials: nm.materials.append(m)
    o=bpy.data.objects.new(name,nm); bpy.context.scene.collection.objects.link(o); return o
pieces=[]
for nm_,mk_ in masks.items():
    o=subset(ped,mk_,nm_); print('alas kaki',nm_,len(o.data.polygons)); pieces.append(o)
bpy.data.objects.remove(ped); objs.remove(ped)

# ---- 4) beri nama & tentukan jatah face ----
plan=[]   # (obj,nama,target_face atau rasio,delimit,onbody)
plan.append((body,'Kursi',BODY_T,{'MATERIAL'},True))
CUSH_T=int(env('CUSHION_TARGET','180000'))
for o in [x for x in objs if x.name.startswith('Meshy_AI_Stitched')]:   # bantalan footrest (kulit) -> ikut lepas/pasang bersama footrest
    b_=bbox(o); side='Kiri' if (b_[0][0]+b_[1][0])/2<XC else 'Kanan'
    plan.append((o,'Footrest_%s_Bantalan'%side,CUSH_T,{'MATERIAL'},True)); objs.remove(o)
for o in pieces: plan.append((o,o.name,FOOT_T if o.name.startswith('Footrest') else PEDAL_BASE_T,{'NORMAL','MATERIAL'},False))
cnt={}; FORCE={}; FORCE_OBJ={}
def nxt(k): cnt[k]=cnt.get(k,-1)+1; return '%s_%02d'%(k,cnt[k])
for o in objs:
    if o is body: continue
    p=par.get(o.name) or ''; n=o.name
    if p=='Assem--KAKI BANGKU SUZUKI': nm='Handrest_Kiri'; r=HAND_R
    elif p=='Assem2': nm='Handrest_Kanan'; r=HAND_R
    elif p=='Assem1--OPSI 3': nm='Handrest_Tengah'; r=HAND_R
    elif p=='Assem--FRAME KAKI': nm='Rangka_Kaki'; r=1.0
    elif n.startswith('rev.3-HANDLE'): nm='Handle'; r=SHELL_R; FORCE['Handle']='part_handle'
    elif n.startswith('Meshy_AI_Orange'): nm='Cupholder'; r=CUP_R; FORCE['Cupholder']='part_cupholder'
    elif p.startswith('Assem-FRAME GAS SPRING'): nm='Sandaran_Belakang'; r=SHELL_R
    elif n=='Tuas': nm='Handrest_Kanan_Tuas'; r=1.0
    else: nm='Rangka_Lain'; r=1.0
    if any(m and 'walnut' in m.name for m in o.data.materials): r=max(r,float(env('WOOD_RATIO','0.4')))
    if 'COVER AR' in n or 'KARET PENAHAN' in n: r=max(r,float(env('HANDCOVER_RATIO','0.6')))   # cover & karet armrest membawa logo Alldila + plat Alpha: jangan dipangkas keras
    ctr=bbox(o); cx=float((ctr[0][0]+ctr[1][0])/2)
    if nm.startswith(('Sandaran_Belakang','Handle','Cupholder')): nm+= '_Kiri' if cx<XC else '_Kanan'
    FORCE_OBJ[o.name]=FORCE.get(nm.rsplit('_',1)[0] if nm.endswith(('_Kiri','_Kanan')) else nm)
    if n.startswith('PUSH BOTTOM'): FORCE_OBJ[o.name]='tuas_merah'   # tombol dorong armrest: seluruhnya merah (bagian hijau penanda ikut merah)
    plan.append((o,nxt(nm) if nm not in ('Handrest_Kanan_Tuas',) else nm,r,{'NORMAL','MATERIAL'},False))

# ---- 5) remap material, decimate, smooth ----
for o,nm,t,dl,onb in plan:
    fm=FORCE_OBJ.get(o.name); o.name=nm; o.data.name=nm; remap_mats(o,onb,fm)
    n0=len(o.data.polygons)
    ratio=(t/n0) if t>1 else t
    if ratio<1 and n0>1500:
        bpy.context.view_layer.objects.active=o
        if o.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')   # blend bisa tersimpan saat objek masih di Edit Mode
        md=o.modifiers.new('dec','DECIMATE'); md.ratio=max(ratio,0.005); md.delimit=dl
        if 'NORMAL' in dl: md.angle_limit=math.radians(25)
        bpy.ops.object.modifier_apply(modifier='dec')
    if nm.startswith('Handrest'):   # halus per sudut: tepi tajam (logo, lekuk cover) tetap tegas, bidang lengkung tetap mulus
        bm_=bmesh.new(); bm_.from_mesh(o.data)
        for e_ in bm_.edges:
            if len(e_.link_faces)==2 and e_.calc_face_angle(0.0)>math.radians(40): e_.smooth=False
        for f_ in bm_.faces: f_.smooth=True
        bm_.to_mesh(o.data); bm_.free()
    else:
        for pl in o.data.polygons: pl.use_smooth=True
    while len(o.data.color_attributes): o.data.color_attributes.remove(o.data.color_attributes[0])   # warna vertex bawaan Meshy ikut mengalikan tint di three.js -> buang
    print('objek',nm,n0,'->',len(o.data.polygons),[m.name for m in o.data.materials])
print('face akhir',sum(len(o.data.polygons) for o,_,_,_,_ in plan))
for m in [m for m in bpy.data.materials if m.users==0]: bpy.data.materials.remove(m)
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,
    export_draco_mesh_compression_level=6,export_materials='EXPORT',export_image_format='AUTO')
print('done',OUT,os.path.getsize(OUT)/1e6)
