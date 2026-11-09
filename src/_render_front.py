"""
Render tampilan depan model gabungan (1 material, belum diklasifikasi) buat
dipakai split_twin_seats.py sebagai acuan spasial yang valid (lihat
penjelasan di split_twin_seats.py kenapa analisis langsung di tekstur UV
tidak bisa dipakai -- UV atlas-nya acak).

Jalankan lewat Blender:
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python src/_render_front.py
"""
import bpy, os, math, mathutils, json

SRC = os.path.abspath("_tmp_merged_nodraco.glb")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
obj = meshes[0]

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 2000
scene.render.resolution_y = 2000
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.08, 0.09, 0.1, 1)
bg.inputs[1].default_value = 1.2  # ambient fill supaya shadow tidak terlalu keras

minv = mathutils.Vector((1e9,) * 3)
maxv = mathutils.Vector((-1e9,) * 3)
for v in obj.bound_box:
    wv = obj.matrix_world @ mathutils.Vector(v)
    minv.x = min(minv.x, wv.x); minv.y = min(minv.y, wv.y); minv.z = min(minv.z, wv.z)
    maxv.x = max(maxv.x, wv.x); maxv.y = max(maxv.y, wv.y); maxv.z = max(maxv.z, wv.z)
center = (minv + maxv) / 2
size = (maxv - minv).length

for rot, energy in [((55, 0, 20), 1.5), ((-40, 0, -30), 0.8)]:
    light = bpy.data.lights.new("sun", type='SUN'); light.energy = energy
    light_obj = bpy.data.objects.new("sun", light)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = tuple(math.radians(a) for a in rot)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = 'FOV'
cam_data.angle = math.radians(35)
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

d = mathutils.Vector((0, -1, 0.12)).normalized()
dist = size * 1.15
cam_obj.location = center + d * dist
direction = (center - cam_obj.location)
cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

scene.render.filepath = os.path.abspath("_tmp_front.png")
bpy.ops.render.render(write_still=True)

cam_info = {
    "loc": list(cam_obj.location),
    "rot_euler": list(cam_obj.rotation_euler),
    # matrix_world eksplisit (bukan dihitung ulang dari euler angle di numpy)
    # supaya tidak ada ambiguitas konvensi rotasi Blender (intrinsic XYZ vs
    # extrinsic) -- row-major 4x4, dipakai langsung sbg forward/right/up.
    "matrix_world": [list(row) for row in cam_obj.matrix_world],
    "angle": cam_data.angle,
    "res": [scene.render.resolution_x, scene.render.resolution_y],
}
with open(os.path.abspath("_tmp_cam.json"), "w") as f:
    json.dump(cam_info, f)
print("CAM", cam_info)
print("DONE")
