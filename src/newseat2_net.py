# Pasang ulang jaring (assets/net-pocket.glb) pada cangkang belakang seat Meshy baru.
# Cangkang baru lebih tebal & lebih tegak, jadi jaring digeser/diputar mengikuti permukaannya (selisih 0,075 m di belakang permukaan, seperti pada cangkang lama).
# Jalankan: blender --background --python src/newseat2_net.py -- <model.glb> <out_net.glb>
import bpy,numpy as np,math,sys,os
P='C:/Users/LENOVO LOQ/Projects/alldila-jok-simulator/'
a=sys.argv[sys.argv.index('--')+1:]; MODEL=a[0]; OUT=a[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=MODEL); bpy.ops.import_scene.gltf(filepath=P+'assets/net-pocket.glb')
def W(o):
    M=np.array(o.matrix_world); v=np.zeros(len(o.data.vertices)*3); o.data.vertices.foreach_get('co',v); return v.reshape(-1,3)@M[:3,:3].T+M[:3,3]
def setW(o,w):
    o.parent=None; o.matrix_world.identity(); o.data.vertices.foreach_set('co',w.ravel()); o.data.update()
OFF=0.075; OLD_SLOPE=0.43; NEW_SLOPE=0.265
phi=math.atan2(1,NEW_SLOPE)-math.atan2(1,OLD_SLOPE)   # (dy,dz): kemiringan cangkang lama -> baru
c,s_=math.cos(phi),math.sin(phi)
for side in ('Kiri','Kanan'):
    sh=W(bpy.data.objects['Sandaran_Belakang_%s_02'%side])
    nets=[o for o in bpy.data.objects if o.name.startswith('Jaring_'+side)]
    ws={o.name:W(o) for o in nets}
    allv=np.concatenate(list(ws.values())); allv[:,2]-=0.1167
    cx=(allv[:,0].min()+allv[:,0].max())/2; cz=(allv[:,2].min()+allv[:,2].max())/2; cy=(allv[:,1].min()+allv[:,1].max())/2
    # tengah panel baru: titik tengah horizontal cangkang, tinggi sama dgn jaring
    xc_new=(sh[:,0].min()+sh[:,0].max())/2 + (cx-(-0.85 if side=='Kiri' else 0.53))   # pertahankan posisi relatif seperti pada cangkang lama
    m=(abs(sh[:,0]-xc_new)<0.1)&(abs(sh[:,2]-cz)<0.04); surf=sh[m,1].max()
    ty=surf+OFF+ (cy-(-0.40))*0   # pusat jaring berada OFF di belakang permukaan
    print('SIDE',side,'cx',cx,'xc_new',xc_new,'surf',surf,'cy',cy,'->',ty,flush=True)
    rot={}
    for o in nets:
        w=ws[o.name].copy(); w[:,2]-=0.1167
        dy=w[:,1]-cy; dz=w[:,2]-cz
        w[:,1]=cy+dy*c-dz*s_
        w[:,2]=cz+dy*s_+dz*c
        rot[o.name]=w
    allr=np.concatenate(list(rot.values())); zmin=allr[:,2].min(); xm=(allr[:,0].min()+allr[:,0].max())/2
    SX=float(os.environ.get('SX','0.96')); SZ=float(os.environ.get('SZ','1.30')); SHX=float(os.environ.get('SHX_'+side,'0'))
    for o in nets:
        w=rot[o.name]
        w[:,0]=xm+(w[:,0]-xm)*SX; w[:,2]=zmin+(w[:,2]-zmin)*SZ
        w[:,0]+=xc_new-cx+SHX; w[:,1]+=ty-cy
        setW(o,w)
for o in list(bpy.data.objects):
    if not o.name.startswith('Jaring_'): bpy.data.objects.remove(o,do_unlink=True)
for m in [m for m in bpy.data.materials if m.users==0]: bpy.data.materials.remove(m)
bpy.ops.export_scene.gltf(filepath=OUT,export_format='GLB',export_draco_mesh_compression_enable=True,export_materials='EXPORT',export_image_format='AUTO')
print('DONE',OUT,os.path.getsize(OUT)/1e3,'KB',flush=True)
