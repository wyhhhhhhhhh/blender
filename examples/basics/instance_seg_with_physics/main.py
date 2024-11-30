import blenderproc as bproc
import argparse
import os
import sys
import random
import numpy as np
import math 

sys.path.append(os.path.dirname(__file__))

from rendering_with_score import *

def extract_bbox_size(bbox):
    xs = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
    ys = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
    zs = np.max(bbox[:, 2]) - np.min(bbox[:, 2])
    return [xs, ys, zs]

def grid_sampler(m, n, k, m_random=True, n_random=True, k_random=True):
    m_s = random.randint(1, m) if m_random else m
    n_s = random.randint(1, n) if n_random else n
 #   m_s = m
 #   n_s = n
    mn = np.zeros(shape=(m_s, n_s))
    count=0
    for m_i in range(m_s):
        for n_i in range(n_s):
            k_s = random.randint(0, k) if k_random else k
         #   k_s = k
            mn[m_i][n_i] = k_s
            count += k_s
    print([m, n, k])
    print([m_s, n_s, k])
    print(mn)
    return mn, count

def grid_pos_generator(mnk, xyz_bbox, xyz_delta):
    x_range =np.linspace(start=0, stop=(xyz_bbox[0]+xyz_delta[0])*mnk[0], num=mnk[0], endpoint=False) + xyz_bbox[0]/2
    y_range =np.linspace(start=0, stop=(xyz_bbox[1]+xyz_delta[1])*mnk[1], num=mnk[1], endpoint=False)+ xyz_bbox[1]/2
    z_range =np.linspace(start=0, stop=(xyz_bbox[2]+xyz_delta[2])*mnk[2], num=mnk[2], endpoint=False)+ xyz_bbox[2]

    x_center = (xyz_bbox[0]+xyz_delta[0])*mnk[0] / 2
    y_center = (xyz_bbox[1]+xyz_delta[1])*mnk[1] / 2
    x_range -= x_center
    y_range -= y_center

    pos = np.zeros(shape=(mnk[0], mnk[1], mnk[2], 3))
    for x_i in range(len(x_range)):
        for y_i in range(len(y_range)):
            for z_i in range(len(z_range)):
                pos[x_i][y_i][z_i] = [x_range[x_i], y_range[y_i], z_range[z_i]]
    
    # position for scene bbox
    up_pos = (y_center+0.1)
    down_pos = -(y_center+0.1)
    left_pos = -(x_center+0.1)
    right_pos = (x_center+0.1)

    return pos, [up_pos, down_pos, left_pos, right_pos]


# give all objs the same angle
def init_rot(objs, euler_angle, perturbation):
    for obj in objs:
        noise=[0,0,0]
        #euler_angle = sample_xyz_pose(angle_xs, angle_ys, angle_zs)
        if perturbation and random.choice([True, False]):
            delta = 3.14/12
            noise = [random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta)]
        obj.set_rotation_euler([x + y for x, y in zip(euler_angle, noise)])

# give objs the pose by grid and bbox
def init_pos(objs, xyz_bbox, xyz_delta, grid, mnk, perturbation):
    pos, scene_pos = grid_pos_generator([grid.shape[0], grid.shape[1], mnk[2]], xyz_bbox, xyz_delta)
    pt = list()
    for m_i in range(grid.shape[0]):
        for n_i in range(grid.shape[1]):
            for k_i in range(int(grid[m_i][n_i])):
                noise=[0,0,0]
                if perturbation and random.choice([True, False]):
                    delta = 0.05
                    noise = [random.uniform(0, delta), random.uniform(0, delta), random.uniform(0, delta)]
                pt.append([x + y for x, y in zip(pos[m_i][n_i][k_i], noise)])

    print("objs:", len(objs))
    for idx, _ in enumerate(objs):
        objs[idx].set_location(pt[idx])	
    return scene_pos


def init_poses(objs, euler_angle, grid, mnk, xyz_delta,  perturbation=False):
    if (np.max(grid) == 0):
        return None

    # set euler angles
    init_rot(objs, euler_angle, False)

    # aabb in current world
    # but from github issue, get_bound_box() returns obb
    # https://github.com/DLR-RM/BlenderProc/issues/444
    bbox = objs[0].get_bound_box() 
    bbox = extract_bbox_size(bbox)
    print(bbox)

    # set euler angles
    init_rot(objs, euler_angle, perturbation)

    xyz_noise = [0,0,0]
    if perturbation and random.choice([True, False]):
        xyz_noise = [random.uniform(0, xyz_delta[0]), random.uniform(0, xyz_delta[1]), random.uniform(0, xyz_delta[2])]
    scene_pos = init_pos(objs, bbox, xyz_noise, grid, mnk, perturbation)

#    # set pos
#    scene_pos = init_pos(objs, bbox, xyz_delta, grid, mnk, perturbation)

    return scene_pos


parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
parser.add_argument('bop_dataset_mm', nargs='?', help="Whether unit of BOP dataset model is mm or not")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('obj_id', nargs='?', default="1", help="Object id for physics positioning")
parser.add_argument('grid_x', nargs='?', default="1", help="Grid x number of objects for physics positioning")
parser.add_argument('grid_y', nargs='?', default="1", help="Grid y number of objects for physics positioning")
parser.add_argument('grid_z', nargs='?', default="1", help="Grid z number of objects for physics positioning")
parser.add_argument('grid_x_delta', nargs='?', default="1", help="Grid x delta of objects for physics positioning")
parser.add_argument('grid_y_delta', nargs='?', default="1", help="Grid y delta of objects for physics positioning")
parser.add_argument('grid_z_delta', nargs='?', default="1", help="Grid z delta of objects for physics positioning")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

bproc.init()

m = int(args.grid_x)
n = int(args.grid_y)
k = int(args.grid_z)
grid, obj_count = grid_sampler(m, n, k, random.choice([True, False]), random.choice([True, False]), random.choice([True, False]))
if obj_count == 0:
    exit(0)

x_delta = float(args.grid_x_delta)
y_delta = float(args.grid_y_delta)
z_delta = float(args.grid_z_delta)

print("obj_ids", args.obj_id)
print("mm", args.bop_dataset_mm)

# load a random sample of bop objects into the scene
sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                                  mm2m = True if args.bop_dataset_mm == "True" else False,
                                  obj_ids=[int(args.obj_id)] * obj_count
                                  )
                                


# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(sampled_bop_objs):
    obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    obj.set_shading_mode('auto')
        
    mat = obj.get_materials()[0]
    if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
        grey_col = np.random.uniform(0.1, 0.9)   
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample object poses and check collisions 
#bproc.object.sample_poses(objects_to_sample = sampled_bop_objs,
#                        sample_pose_func = sample_pose_func, 
#                        max_tries = 1000)

def meshgrid2(x_a, y_a, z_a):
    x_n = len(x_a)
    y_n = len(y_a)
    z_n = len(z_a)
    print(x_n, y_n, z_n)
    xyz = np.zeros(shape=(x_n, y_n, z_n, 3))
    for i in range(x_n):
        for j in range(y_n):
            for k in range(z_n):
                xyz[i,j,k] = np.asarray([x_a[i], y_a[j], z_a[k]])
    return xyz


def sample_grid_pose(sampled_bop_objs, x_n, y_n, z_n, x_d, y_d, z_d, delta=0.001):
    x_pos = np.arange(0, x_n * x_d-delta, x_d)
    y_pos = np.arange(0, y_n * y_d-delta, y_d)
    z_pos = np.arange(0, z_n * z_d-delta, z_d)
    x_pos = x_pos - x_n * x_d / 2 + x_d/2
    y_pos = y_pos - y_n * y_d / 2
    z_pos = z_pos + z_d/2
    print(y_pos)
    print(len(x_pos), len(y_pos), len(z_pos))
    pos = meshgrid2(x_pos, y_pos, z_pos)
    print(pos.shape)

    assert(len(sampled_bop_objs) == (x_n*y_n*z_n))
    pos = pos.reshape(x_n*y_n*z_n, 3)
    print(pos)
    for idx in range(len(pos)):
        sampled_bop_objs[idx].set_location(pos[idx])
        sampled_bop_objs[idx].set_rotation_euler([np.random.uniform(0, 1.570796), 0, 0])

#sample_grid_pose(sampled_bop_objs, 1, 3, 2, 1.2, 0.3, 0.3)


# generate position of objects
#scene_pos = init_poses(sampled_bop_objs, [0,0,0], grid, [m,n,k], [0.05,0.02,0.05], True)

## pose_type and angle should be different for each object
#pose_type = random.randint(2,3)
#if pose_type == 0:
#    euler_angles=[0,0,0]
#elif pose_type == 1:
#    euler_angles=[0,0,1.570796]
#elif pose_type == 2:
#    euler_angles=[3.1415926, 0, 0]
#elif pose_type == 3:
#    euler_angles=[3.1415926, 0, 1.570796]
#elif pose_type == 4:
#    euler_angles=[0,1.570796,0]
#else:
#    euler_angles=[1.570796,0,1.570796]

euler_angles=[0,0,0]
angle_xs = [0, math.pi]
angle_ys = [0, math.pi]
angle_zs = [0, math.pi/2, math.pi, math.pi*3/2]

def sample_xyz_pose(xs, ys, zs):
    euler_angles = [random.choice(xs), random.choice(ys), random.choice(zs)]
    return euler_angles
euler_angles = sample_xyz_pose(angle_xs, angle_ys, angle_zs)

if random.choice([True, False]):
    pertub=True
else:
    pertub=False

scene_pos = init_poses(sampled_bop_objs, euler_angles, grid, [m,n,k], [x_delta,y_delta,z_delta], False)
print("scene_pos", scene_pos)
max_l = max(scene_pos)


neg_y = scene_pos[0]
pos_y = scene_pos[1]
pos_x = scene_pos[2]
neg_x = scene_pos[3]
x_w = (pos_x-neg_x)/2
y_w = (pos_y-neg_y)/2
x_l = 0.5
y_l = 0.5
z_l = 0.05 # z-axis scale not working in blenderproc!

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[x_w, y_w, 1]),
            bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, -y_w, 0], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, y_w, 0], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[x_w, 0, 0], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[-x_w, 0, 0], rotation=[0, 1.570796, 0])]

## create room
#room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
#               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, scene_pos[1], 1], rotation=[-1.570796, 0, 0]),
#               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, scene_pos[0], 1], rotation=[1.570796, 0, 0]),
#               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[scene_pos[3], 0, 1], rotation=[0, -1.570796, 0]),
#               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[scene_pos[2], 0, 1], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

# sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
#random_cc_texture = np.random.choice(cc_textures)
random_cc_texture = cc_textures[0]
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

        
# Physics Positioning
#use_physics = random.choice([True, False])
use_physics = True
if use_physics:
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)


# Here we use fix distance from scene in z-axis for camera pose
cam_position = [0, 0, 2]
cam_rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, -1])
cam_matrix = bproc.math.build_transformation_mat(cam_position, cam_rotation_matrix)
bproc.camera.add_camera_pose(cam_matrix)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# render the whole pipeline
data = bproc.renderer.render()

# Write data in bop format
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                       dataset = args.bop_dataset_name,
                       depths = data["depth"],
                       colors = data["colors"], 
                       color_file_format = "JPEG",
                       ignore_dist_thres = 10)

# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["class", "instance"]))

# write the data to a .hdf5 container
bproc.writer.write_hdf5(os.path.join(args.output_dir, 'bop_data', args.bop_dataset_name, 'train_pbr', '000000'), data)

# Render single objects for complete object mask
render_mask = os.path.join(args.output_dir, 'bop_data', args.bop_dataset_name, 'train_pbr', '000000', "rendering")
os.makedirs(render_mask, exist_ok=True)

for idx in range(len(sampled_bop_objs)):
    render_single_object(sampled_bop_objs, idx, render_mask)

# filter small ratio mask 
h5_mixed_file = os.path.join(args.output_dir, 'bop_data', args.bop_dataset_name, 'train_pbr', '000000', '0.hdf5')
h5_filtered_file = os.path.join(args.output_dir, 'bop_data', args.bop_dataset_name, 'train_pbr', '000000', '0_f.hdf5')
merge_labels(len(sampled_bop_objs), render_mask, h5_mixed_file, h5_filtered_file)

#merge_labels_v2(len(sampled_bop_objs), h5_mixed_file, h5_filtered_file)