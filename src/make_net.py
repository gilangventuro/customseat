# Membuat kantung jaring (net pocket) di cangkang belakang sandaran twin seat -> assets/net-pocket.glb
# Dipasang di antara empat baut soket yang sudah ada di cangkang (posisi diukur dari peta tinggi cangkang).
# Jalankan: blender --background --python src/make_net.py -- assets/twin-seat-meshy.glb assets/net-pocket.glb
# Koordinat = dunia Blender (sama dengan model kursi); GLB hasilnya dipasang sebagai anak model di situs.
import bpy, bmesh, sys, numpy as np, math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

SRC, OUT = sys.argv[-2], sys.argv[-1]
# (nama cangkang, nama sisi, pusat x, pusat z) -- pusat = tengah keempat baut; jarak baut 0.70 x 0.27 m
SEATS = [('Sandaran_Belakang_Kanan_02', 'Kanan', 0.516, 1.157),
         ('Sandaran_Belakang_Kiri_02',  'Kiri', -0.854, 1.188)]
BW, BH = 0.70, 0.27          # jarak baut (lebar x tinggi)
PAD = 0.11                   # bantalan persegi di tiap sudut (menutup bundaran baut di cangkang)
BAND = 0.03                  # lebar pita bingkai
FRAME_TOP, FRAME_T = 0.013, 0.012   # tinggi permukaan atas bingkai dari cangkang, tebal bingkai
NET_D0, NET_BULGE = 0.005, 0.024    # jaring: tinggi di tepi, tonjolan maksimum di tengah

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
dg = bpy.context.evaluated_depsgraph_get()

def mat(name, col, rough):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = col; b.inputs['Roughness'].default_value = rough
    return m
M_NET = mat('net_mesh', (0.04, 0.045, 0.05, 1), 0.9)
M_FRAME = mat('net_frame', (0.02, 0.022, 0.025, 1), 0.8)

def make_surface(shell, cx, cz):
    """permukaan belakang cangkang di sekitar kantung: y = f(x,z) (polinom orde 2, lubang baut diabaikan) + normalnya"""
    bvh = BVHTree.FromObject(shell, dg)
    xs = np.arange(cx - BW/2 - 0.06, cx + BW/2 + 0.06, 0.01); zs = np.arange(cz - BH/2 - 0.06, cz + BH/2 + 0.06, 0.01)
    P = []
    for z in zs:
        for x in xs:
            r = bvh.ray_cast(Vector((x, 0.6, z)), Vector((0, -1, 0)))
            if r[0] is not None: P.append((x, r[0].y, z))
    P = np.array(P); A = lambda x, z: np.stack([np.ones_like(x), x, z, x*x, x*z, z*z], 1)
    keep = np.ones(len(P), bool)
    for _ in range(4):
        c, *_r = np.linalg.lstsq(A(P[keep, 0], P[keep, 2]), P[keep, 1], rcond=None)
        res = np.abs(A(P[:, 0], P[:, 2]) @ c - P[:, 1]); keep = res < max(0.0015, np.percentile(res, 80))
    print('permukaan', shell.name, 'titik', len(P), 'dipakai', int(keep.sum()), 'galat maks (mm)', round(float(res[keep].max()) * 1000, 2))
    def S(x, z):
        y = c[0] + c[1]*x + c[2]*z + c[3]*x*x + c[4]*x*z + c[5]*z*z
        yx = c[1] + 2*c[3]*x + c[4]*z; yz = c[2] + c[4]*x + 2*c[5]*z
        n = Vector((-yx, 1.0, -yz)).normalized()
        return Vector((x, y, z)), n
    return S

def new_obj(name, bm, material):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.materials.append(material)
    for p in me.polygons: p.use_smooth = True
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o); return o

def orient_up(bm, S):
    """pastikan normal face menghadap keluar (searah normal permukaan)"""
    bm.normal_update()
    for f in bm.faces:
        c = f.calc_center_median(); _, n = S(c.x, c.z)
        if f.normal.dot(n) < 0: f.normal_flip()
    bm.normal_update()

built = []
for shell_name, side, cx, cz in SEATS:
    S = make_surface(bpy.data.objects[shell_name], cx, cz)
    Wo, Ho = BW + 0.05, BH + 0.05           # ukuran luar bingkai
    Wi, Hi = Wo - 2*BAND, Ho - 2*BAND       # ukuran dalam (jendela jaring)
    # ---- jaring ----
    nu, nv = 70, 28; Wn, Hn = Wi + 0.008, Hi + 0.008
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new('UVMap'); grid = []
    for j in range(nv + 1):
        row = []
        for i in range(nu + 1):
            u, v = i/nu, j/nv
            x = cx - Wn/2 + u*Wn; z = cz - Hn/2 + v*Hn
            ed = min(u, 1-u, v, 1-v)
            bulge = NET_BULGE * (math.sin(math.pi*u)**0.6) * (math.sin(math.pi*v)**0.6)
            p, n = S(x, z); row.append(bm.verts.new(p + n*(NET_D0 + bulge)))
        grid.append(row)
    for j in range(nv):
        for i in range(nu):
            f = bm.faces.new((grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i]))
            for l, (a, b) in zip(f.loops, ((i, j), (i+1, j), (i+1, j+1), (i, j+1))):
                l[uvl].uv = (a/nu*Wn, b/nv*Hn)      # UV dalam meter (tekstur jaring diulang di situs)
    orient_up(bm, S)
    built.append(new_obj('Jaring_%s_Net' % side, bm, M_NET))
    # ---- bingkai + paku ----
    bm = bmesh.new(); nx, nz = 76, 32; gr = []
    for j in range(nz + 1):
        row = []
        for i in range(nx + 1):
            x = cx - Wo/2 + i/nx*Wo; z = cz - Ho/2 + j/nz*Ho
            p, n = S(x, z); row.append(bm.verts.new(p + n*FRAME_TOP))
        gr.append(row)
    fs = []
    for j in range(nz):
        for i in range(nx):
            xc = cx - Wo/2 + (i + .5)/nx*Wo; zc = cz - Ho/2 + (j + .5)/nz*Ho
            if abs(xc - cx) < Wi/2 and abs(zc - cz) < Hi/2:                   # jendela jaring, kecuali bantalan sudut penutup baut
                if not any(abs(xc - (cx + sx*BW/2)) < PAD/2 and abs(zc - (cz + sz*BH/2)) < PAD/2 for sx in (-1, 1) for sz in (-1, 1)): continue
            fs.append(bm.faces.new((gr[j][i], gr[j][i+1], gr[j+1][i+1], gr[j+1][i])))
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    orient_up(bm, S)
    bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=FRAME_T)
    bm.normal_update()
    built.append(new_obj('Jaring_%s_Bingkai' % side, bm, M_FRAME))

for o in list(bpy.data.objects):
    if o not in built: bpy.data.objects.remove(o)
bpy.ops.object.select_all(action='DESELECT')
for o in built: o.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', use_selection=True, export_apply=True)
print('net: selesai', OUT, [(o.name, len(o.data.polygons)) for o in built])
