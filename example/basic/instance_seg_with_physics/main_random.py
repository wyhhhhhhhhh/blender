import blenderproc as bproc
import argparse
import os
import random
import numpy as np

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('obj_id', nargs='?', default="1", help="Object id for physics positioning")
parser.add_argument('obj_num', nargs='?', default="1", help="Number of objects for physics positioning")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

bproc.init()

# load a random sample of bop objects into the scene
sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                                  mm2m = True,
                                  obj_ids=[int(args.obj_id)] * int(args.obj_num)
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
        
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -0.5, 0.5], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0.5, 0.5], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[1.2, 0, 1.2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-1.2, 0, 1.2], rotation=[0, 1.570796, 0])]
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
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample object poses and check collisions 
bproc.object.sample_poses(objects_to_sample = sampled_bop_objs,
                        sample_pose_func = sample_pose_func, 
                        max_tries = 1000)

## Physics Positioning
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                max_simulation_time=10,
                                                check_object_interval=1,
                                                substeps_per_frame = 20,
                                                solver_iters=25)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)


# Here we use fix distance from scene in z-axis for camera pose
cam_position = [0, 0, 2.3]
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