import blenderproc as bproc
import argparse
import os
import random
import numpy as np
import h5py
import cv2
from PIL import Image
import mathutils
from PIL import Image, ImageOps


from blenderproc.python.types.MaterialUtility import Material

# 是否只生成未遮挡物体
unoccupied = False

def rectify_stereo_images(left_rgb_image, right_rgb_image, left_depth, right_depth, K, R, t):
    # # Load your RGB image and depth map
    # left_rgb_image = cv2.imread(left_image)
    # right_rgb_image = cv2.imread(right_image)
    # left_depth_map = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED)

    distCoeff = np.zeros(4)
    # Rectify RGB image
    print(K)
    print(left_rgb_image.shape[:2])
    print(R)
    print(t)
    w, h = left_rgb_image.shape[1], left_rgb_image.shape[0]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, distCoeff, K, distCoeff, [w, h], R.T, -R.T@t)
    map1x, map1y = cv2.initUndistortRectifyMap(K, distCoeff, R1, P1, [w, h], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, distCoeff, R2, P2, [w, h], cv2.CV_32FC1)
    rectified_left_rgb = cv2.remap(left_rgb_image, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    rectified_right_rgb = cv2.remap(right_rgb_image, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    rectified_left_depth = cv2.remap(left_depth, map1x, map1y, interpolation=cv2.INTER_NEAREST)
    rectified_right_depth = cv2.remap(right_depth, map2x, map2y, interpolation=cv2.INTER_NEAREST)
    print(np.min(left_depth), np.max(left_depth))
    print(np.min(rectified_left_depth), np.max(rectified_left_depth))
    print(rectified_left_depth)
    count = np.sum((rectified_left_depth > 0) & (rectified_left_depth < 0.8))
    print(count)

    f = Q[2][3]
    q43 = Q[3][2]
    q44 = Q[3][3]
    print(Q)

    left_disparity = np.zeros_like(rectified_left_depth)
    left_disparity[rectified_left_depth > 0] = (f / rectified_left_depth[rectified_left_depth > 0] - q44) / q43 
    print(left_disparity.shape)
    print(np.min(left_disparity), np.max(left_disparity))
    print(left_disparity)

    # Display rectified images
    from matplotlib import pyplot as plt
    import matplotlib
    fig, ax = plt.subplots(nrows=4, ncols=2)
    plt.subplot(4, 2, 1)
    plt.imshow(left_rgb_image)
    plt.subplot(4, 2, 2)
    plt.imshow(right_rgb_image)
    plt.subplot(4, 2, 3)
    plt.imshow(rectified_left_rgb)
    plt.subplot(4, 2, 4)
    plt.imshow(rectified_right_rgb)
    plt.subplot(4, 2, 5)
    plt.imshow(rectified_left_depth)
    plt.subplot(4, 2, 6)
    plt.imshow(rectified_right_depth)
    plt.subplot(4, 2, 7)
    plt.imshow(left_disparity)
    plt.subplot(4, 2, 8)
    plt.imshow(left_disparity)
    # plt.show(block=False)
    plt.savefig("mygraph.png")
    # plt.pause(10)
    plt.close()

    return rectified_left_rgb, rectified_right_rgb, rectified_left_depth, rectified_right_depth, left_disparity, Q

def write_hdf5_with_scores(hdf5_path, label, scores):
    h5f = h5py.File(hdf5_path, 'w')
    h5f.create_dataset('instance_segmaps', data=label)

    dict_group = h5f.create_group('instance_scores')
    for k, v in scores.items():
        dict_group[k] = v
    #h5f.create_dataset('instance_scores', data=scores)
    h5f.close()

def write_hdf5(hdf5_path, label):
    h5f = h5py.File(hdf5_path, 'w')
    h5f.create_dataset('instance_segmaps', data=label)
    h5f.close()

def read_hdf5(hdf5_path):
    filename = hdf5_path
    with h5py.File(filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        label_img = np.array(f["instance_segmaps"][:])
       # print(np.max(label_img))
       # print(np.min(label_img))
        return label_img


def disparity_to_point_cloud(disparity_map, Q):
    """
    Convert disparity map to point cloud.

    Args:
    - disparity_map: np.array, the disparity map
    - Q: np.array, the 4x4 reprojection matrix obtained from stereo rectification

    Returns:
    - points_3D: np.array, the 3D points cloud
    """
    # Reproject points to 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    
    # Mask to filter out points with invalid disparity (usually set to 0)
    mask = disparity_map > disparity_map.min()
    points_3D = points_3D[mask]
    
    return points_3D

def show_points(points):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("show.ply", pcd)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    #vis.get_render_option().background_color = [0,0,0]
    vis.run()
    vis.destroy_window()

def read_stereo_hdf5(left_hdf5_path, right_hdf5_path, scene_camera_path):
    left_filename = left_hdf5_path
    right_filename = right_hdf5_path
    with h5py.File(left_filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        left_image = np.array(f["colors"][:])
        left_depth = np.array(f["depth"][:])
        print(left_image.shape)
        print(left_depth.shape)
       # print(np.max(label_img))
       # print(np.min(label_img))
    with h5py.File(right_filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        right_image = np.array(f["colors"][:])
        right_depth = np.array(f["depth"][:])
    import json
    with open(scene_camera_path) as f:
        scene_camera = json.load(f)
        K = np.array(scene_camera["0"]["cam_K"]).reshape(3,3)
        left_R = np.array(scene_camera["0"]["cam_R_w2c"]).reshape(3,3)
        left_t = np.array(scene_camera["0"]["cam_t_w2c"]) * 0.001
        right_R = np.array(scene_camera["1"]["cam_R_w2c"]).reshape(3,3)
        right_t = np.array(scene_camera["1"]["cam_t_w2c"]) * 0.001
        left_T = np.identity(4)
        left_T[:3, :3] = left_R
        left_T[:3, 3] = left_t
        print(left_T)
        inv_right_T = np.identity(4)
        inv_right_T[:3, :3] = right_R.T
        inv_right_T[:3, 3] = -right_R.T @ right_t
        print(inv_right_T)
        T = left_T @ inv_right_T
        R = T[:3,:3]
        t = T[:3, 3]
    return left_image, right_image, left_depth, right_depth, K, R, t

def save_pfm(filename, image, scale=1):
    """Save a 2D NumPy array as a PFM file.
    
    Args:
        filename (str): The file path to save the PFM file.
        image (np.ndarray): The 2D array to save.
        scale (float): Scale factor for the image values. 
                       Typically 1 or -1 for little-endian or big-endian respectively.
    """
    if image.dtype != np.float32:
        raise ValueError('Image dtype must be float32.')
    
    if len(image.shape) == 2:
        color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    else:
        raise ValueError('Image must be a HxW or HxWx3 array.')

    file = open(filename, 'wb')

    if color:
        file.write(b'PF\n')
    else:
        file.write(b'Pf\n')
        
    file.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())

    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and np.little_endian:
        scale = -scale

    file.write(f'{scale}\n'.encode())
    
    image.tofile(file)
    file.close()


def rectify_stereo_scene(root_path, sim_id, scene_id, mode, output_path):
    cur_path = os.path.join(root_path, "sim"+sim_id, mode, "Scene_{}".format(scene_id), "bop_data", "lm", "train_pbr", "000000")
    left_h5 = os.path.join(cur_path, "0.hdf5")
    right_h5 = os.path.join(cur_path, "1.hdf5")
    scene_camera = os.path.join(cur_path, "scene_camera.json")

    left_image, right_image, left_depth, right_depth, K, R, t = read_stereo_hdf5(left_h5, right_h5, scene_camera)
    rectified_left_rgb, rectified_right_rgb, rectified_left_depth, rectified_right_depth, left_disparity, Q = rectify_stereo_images(left_image, right_image, left_depth, right_depth, K, R, t)
    # disparity map

    # save all to output
    simid_disparity_folder = os.path.join(output_path, "sim"+sim_id+"_disparity")
    scene_folder = os.path.join(simid_disparity_folder, "scene")
    disparity_left = os.path.join(scene_folder, "left")
    disparity_right = os.path.join(scene_folder, "right")
    if not os.path.exists(disparity_left):
        os.makedirs(disparity_left, exist_ok=True)
    if not os.path.exists(disparity_right):
        os.makedirs(disparity_right, exist_ok=True)

    simid_cleanpass_folder = os.path.join(output_path, "sim"+sim_id+"_frames_cleanpass")
    scene_folder = os.path.join(simid_cleanpass_folder, "scene")
    cleanpass_left = os.path.join(scene_folder, "left")
    cleanpass_right = os.path.join(scene_folder, "right")
    if not os.path.exists(cleanpass_left):
        os.makedirs(cleanpass_left, exist_ok=True)
    if not os.path.exists(cleanpass_right):
        os.makedirs(cleanpass_right, exist_ok=True)
    
    left_image = Image.fromarray(rectified_left_rgb)
    right_image = Image.fromarray(rectified_right_rgb)
    
    left_image.save(os.path.join(cleanpass_left, f'{int(scene_id):04d}.png'))
    right_image.save(os.path.join(cleanpass_right, f'{int(scene_id):04d}.png'))

    save_pfm(os.path.join(disparity_left, f'{int(scene_id):04d}.pfm'), left_disparity)

    show_points(disparity_to_point_cloud(left_disparity, Q))



def render_single_object(objs, render_idx, h5_folder):
    h5_folder = os.path.join(h5_folder, str(render_idx))
    if not os.path.exists(h5_folder):
        os.mkdir(h5_folder)

    for idx, obj in enumerate(objs):
        if idx != render_idx:
            obj.hide()
    
    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(h5_folder, 'bop_data'),
                        dataset = 'lm',
                        depths = data["depth"],
                        colors = data["colors"], 
                        color_file_format = "JPEG",
                        ignore_dist_thres = 10)

    # Render segmentation masks (per class and per instance)
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance"]))

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5(h5_folder, data)

    # recover rendering
    for idx, obj in enumerate(objs):
            obj.hide(False)

def sample_gray_value(category):
    if category == "dark":
        base_value = random.randint(0, 50)  # Dark range (closer to black)
    elif category == "light":
        base_value = random.randint(80, 230)  # Light range (closer to white)
    else:
        raise ValueError("Unknown category")
    
    # Apply minor random variation, ensuring it stays within 0-255 range
    variation = random.randint(-10, 10)
    gray_value = max(0, min(255, base_value + variation)) / 255

    return gray_value

def sample_gray():
    c = ["dark", "light"]
    return sample_gray_value(random.sample(c, 1)[0])


def create_material():
    import bpy
    # Create a new material
    material = bpy.data.materials.new(name="Principled_BSDF_Material")
    material.use_nodes = True  # Enable node-based material

    # Get the material's node tree
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear all nodes to start fresh
    for node in nodes:
        nodes.remove(node)

    # Add a Principled BSDF node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)

    # Set the base color of the Principled BSDF node
    bsdf_node.inputs['Base Color'].default_value = (1, 0, 0, 1)  # Red color (RGBA)
    bsdf_node.inputs['Metallic'].default_value = 0.0
    bsdf_node.inputs['Roughness'].default_value = 0.0
    bsdf_node.inputs['Transmission'].default_value = 0.0
    bsdf_node.inputs['Alpha'].default_value = 1.0
    bsdf_node.inputs['IOR'].default_value = 1.5

    # Add a Material Output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    # Link the BSDF node to the Material Output node
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    return material

def load_images(image_folder):
    import glob
    # Get all image files in the folder with full paths
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))

    # Print the full paths of all image files
    image_list = []
    for image_path in image_paths:
        image_list.append(os.path.abspath(image_path))
    return image_list

def create_random_background(image_list, tiled=False, pbr=False):
    import bpy

    material = bpy.data.materials.new(name="TiledImageMaterial")
    material.use_nodes = True  # Enable nodes for the material

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    texture_node = nodes.new(type='ShaderNodeTexImage')
    mapping_node = nodes.new(type='ShaderNodeMapping')
    texture_coord_node = nodes.new(type='ShaderNodeTexCoord')

    # Load an image
    image_length = len(image_list)
    image_idx = np.random.randint(0, image_length)
    image_path = image_list[image_idx]
    image = bpy.data.images.load(image_path)
    texture_node.image = image

    # Connect the nodes
    links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])
    links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Step 9: Adjust the Mapping node to repeat the texture
    if tiled:
        # this value is related to plane size, assume the plane size=2
        scale_max = 5
        location_max = 5
        rotation_max = 3.14/4
        scale_x = np.random.uniform(1, scale_max)
        scale_y = np.random.uniform(1, scale_max)
        location_x = np.random.uniform(-location_max, location_max)
        location_y = np.random.uniform(-location_max, location_max)
        rotation_x = np.random.uniform(-rotation_max, rotation_max)
        rotation_y = np.random.uniform(-rotation_max, rotation_max)
        rotation_z = np.random.uniform(-rotation_max, rotation_max)
        mapping_node.inputs['Scale'].default_value = (scale_x, scale_y, 1)  # Adjust the scale for repeating
        mapping_node.inputs['Location'].default_value = (location_x, location_y, 0)  
        mapping_node.inputs['Rotation'].default_value = (rotation_x, rotation_y, rotation_z)  
    
    if pbr:
        principled_node.inputs['Roughness'].default_value = np.random.uniform(0, 1)
        principled_node.inputs['Metallic'].default_value = np.random.uniform(0, 1)

    return material


def uniform_sample_excluding_subrange(a, b, c, d):
    if c <= a or d >= b or c >= d:
        print("Invalid subrange")
        return np.random.choice([a, d])
    
    # Calculate lengths of the segments
    length1 = c - a
    length2 = b - d
    total_length = length1 + length2
    
    # Decide which segment to sample from
    if random.uniform(0, total_length) < length1:
        # Sample from [a, c)
        return random.uniform(a, c)
    else:
        # Sample from (d, b]
        return random.uniform(d, b)

# scale is radius of bbox in blender for primitive
# scale=1, means 2m in length
def add_random_obj(bg_images, x_bound=0.1, y_bound=0.1, z_bound=0, max_num=10):
    def add_one():
        primitives = ['CUBE', 'SPHERE', 'CONE', 'CYLINDER']

        max_scale = 0.15 # max_length=2*max_scale
        max_loc = 0.4
        max_rot = 3.14/3
        scale_x = np.random.uniform(max_scale//2, max_scale)
        scale_y = np.random.uniform(max_scale//2, max_scale)
        scale_z = np.random.uniform(max_scale//2, max_scale)
        scale_xyz = max(scale_x, scale_y, scale_z)
        location_x = uniform_sample_excluding_subrange(-max_loc, max_loc, -x_bound-scale_xyz, x_bound+scale_xyz)
        location_y = uniform_sample_excluding_subrange(-max_loc, max_loc, -y_bound-scale_xyz, y_bound+scale_xyz)
        location_z = np.random.uniform(z_bound, 1)
        rotation_x = np.random.uniform(-max_rot, max_rot)
        rotation_y = np.random.uniform(-max_rot, max_rot)
        rotation_z = np.random.uniform(-max_rot, max_rot)

        obj = bproc.object.create_primitive(np.random.choice(primitives), scale=[scale_x, scale_y, scale_z], location=[location_x, location_y, location_z], rotation=[rotation_x, rotation_y, rotation_z])
        obj.enable_rigidbody(True, collision_shape='CONVEX_HULL', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        cur_mat = create_random_background(bg_images, tiled=False, pbr=True)
        obj.replace_materials(Material(cur_mat))
        obj.set_cp("category_id", 0)

        # print("FFF: ", [location_x, location_y, location_z])
        # print("FFFFFF: ", [scale_x, scale_y, scale_z])
    
    for _ in range(np.random.randint(max_num//2, max_num)):
        add_one()


def add_random_bg_planes(bg_images, max_num=5):
    # def add_one(bottom_angle = 3.14/6):
    def add_one(bottom_angle = 0):    
        bottom_locs = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0]
        bottom_rots = [np.random.uniform(-bottom_angle, bottom_angle), np.random.uniform(-bottom_angle, bottom_angle), np.random.uniform(-bottom_angle, bottom_angle)]
        bottom_plane = bproc.object.create_primitive('PLANE', scale=[2, 2, 0.01], location=bottom_locs, rotation=bottom_rots)
        bottom_plane.enable_rigidbody(False, collision_shape='CONVEX_HULL', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        cur_mat = create_random_background(bg_images, tiled=True)
        bottom_plane.replace_materials(Material(cur_mat))
        bottom_plane.set_cp("category_id", 0)
    
    add_one()
    # cur_num = np.random.randint(0, max_num)
    # for _ in range(cur_num):
    #     add_one()

# set shading and physics properties and randomize PBR materials
def random_color_mat(obj):
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1))
    # mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 1.0))
    # mat.set_principled_shader_value("Base Color", [np.random.uniform(0, 1.0), np.random.uniform(0, 1.0), np.random.uniform(0, 1.0), 1])        

    # blue box
    mat.set_principled_shader_value("Base Color", [np.random.uniform(0, 1.0), np.random.uniform(0, 1), np.random.uniform(0, 1.0), 1])        
    # mat.set_principled_shader_value("Base Color", [0.015,0.015,0.015, 1])        
    # mat.set_principled_shader_value("IOR", 2.0)        
    # mat.set_principled_shader_value("Alpha", 0.4)        

def set_obj_material(obj, transparent=False):
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]

    if not transparent:
        # non-transparent objects
        if np.random.choice([True, False]):
        # if True:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.1))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 0.1))
        else:
            random_color_mat(obj)
    else:
        # transparent objects
        if np.random.choice([True, False]):
        # if True:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.1))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 0.1))
            mat.set_principled_shader_value("Base Color", [1.0,1.0,1.0, 1])        
            mat.set_principled_shader_value("Transmission", 1.0)        
            mat.set_principled_shader_value("Emission Strength", 0.0)
            mat.set_principled_shader_value("IOR", np.random.uniform(1.49, 1.59))        
        else:
            # random_color_mat(obj)
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.2))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 0.2))
            mat.set_principled_shader_value("Base Color", [np.random.uniform(0, 1),np.random.uniform(0, 1),np.random.uniform(0, 1), 1])        
            mat.set_principled_shader_value("Transmission", 1.0)        
            mat.set_principled_shader_value("Emission Strength", 0.0)
            mat.set_principled_shader_value("Alpha", np.random.uniform(0.6, 1))
            mat.set_principled_shader_value("IOR", np.random.uniform(1.49, 1.59))        

def set_obj_poses(sampled_bop_objs, x_w, y_w, w_delta):
    import math
    def sample_xyz_pose(xs, ys):
        euler_angles = [random.choice(xs), random.choice(ys), random.uniform(0, 2*math.pi)]
        return euler_angles
    #euler_angles = sample_xyz_pose(angle_xs, angle_ys, angle_zs)

    # Define a function that samples 6-DoF poses
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.array([-(x_w+w_delta), -(y_w+w_delta), 0.2])
        max = np.array([(x_w+w_delta), (y_w+w_delta), 1.0])
        pos = np.random.uniform(min, max)
        print("sampled pos:", pos)
        # eulers = sample_xyz_pose(angle_xs, angle_ys)
        # print("sampled euler angles:", eulers)
        obj.set_location(pos)
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        # obj.set_rotation_euler([0, 0, 0])
        # obj.set_rotation_euler(eulers)

    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)

def random_color_mat_plane(obj, roughness, metallic, base_color):
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Metallic", metallic)
    mat.set_principled_shader_value("Base Color", base_color)        

def create_box(bg_images, cc_textures):
    # make box same as real scene
    # create room
    # x_l = 0.1
    # x_l = np.random.uniform(0.11, 0.13) # half length
    x_l = np.random.uniform(0.1, 0.15) # half length
    # y_l = np.random.uniform(0.1, 0.15)
    y_l = x_l
    # z_l = 0.01 # z-axis scale not working in blenderproc!
    z_l = np.random.uniform(0.005, 0.01) # z-axis scale not working in blenderproc!
    # x_w = args.box_width
    # y_w = args.box_width / 1.5 
    # x_w = np.random.uniform(args.box_width/3, args.box_width)
    # y_w = np.random.uniform(args.box_width/3, args.box_width)

    # x_w = np.random.uniform(0.33/2, 0.35/2)
    # y_w = np.random.uniform(0.24/2, 0.26/2)

    # old
    x_w = np.random.uniform(0.3/2, 0.4/2)
    y_w = np.random.uniform(0.2/2, 0.3/2)
    # new
    x_w = np.random.uniform(0.33/2, 0.37/2)
    y_w = np.random.uniform(0.23/2, 0.27/2)

    w_delta = -0.1
    room_planes = [bproc.object.create_primitive('CUBE', scale=[x_w, y_w, 0.001]),
                bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, -y_w, 0], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, y_w, 0], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[x_w, 0, 0], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[-x_w, 0, 0], rotation=[0, 1.570796, 0])]

    # add material
    if np.random.choice([True, False]):
    # if True:
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            plane.replace_materials(random_cc_texture)
            random_color_mat(plane)
            plane.set_cp("category_id", 2)
            # print(plane.get_name())
            # print(plane.get_cp("category_id"))
    else:
        for plane in room_planes:
            plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            if np.random.choice([True, False]):
                cur_mat = create_random_background(bg_images, tiled=True)
                plane.replace_materials(Material(cur_mat))
            else:
                roughness = np.random.uniform(0,1)
                metallic = np.random.uniform(0,1)
                base_color = sample_gray()
                # base_colors = [np.random.uniform(0, 0.2), np.random.uniform(0, 0.2), np.random.uniform(0, 0.2), 1]
                base_colors = [base_color, base_color, base_color, 1]
                plane.new_material("box")
                random_color_mat_plane(plane, roughness, metallic, base_colors)
            plane.set_cp("category_id", 2)
            # print(plane.get_name())
            # print(plane.get_cp("category_id"))                  
    return x_w, y_w, w_delta

def add_random_objs_with_box(sampled_bop_objs, bg_images, cc_textures, transparent=False):
    for j, obj in enumerate(sampled_bop_objs):
        obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.set_shading_mode('auto')
        set_obj_material(obj, transparent=transparent)
        obj.set_cp("category_id", 1)
        # print(obj.get_name())
    return create_box(bg_images, cc_textures)

def update_objs(sampled_bop_objs, bg_images, cc_textures, transparent=False):
    for j, obj in enumerate(sampled_bop_objs):
        obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99, mass = 0.0)
        if j + 1 in selected_list:
            pass
        else:
            obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.set_shading_mode('auto')
            set_obj_material(obj, transparent=transparent)
            obj.set_cp("category_id", 1)
            obj.set_location([1000, 1000, 1000])
    # create_box(bg_images, cc_textures)

def add_random_objs_without_box(sampled_bop_objs, bg_images, cc_textures, transparent=False):
    for j, obj in enumerate(sampled_bop_objs):
        obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)  # whether to enable physics engine
        obj.set_shading_mode('auto')
        set_obj_material(obj, transparent=transparent)
        obj.set_cp("category_id", 1)
        # print(obj.get_name())
    return 

def add_random_box(bg_images, cc_textures):
    return create_box(bg_images, cc_textures)

def hide_sampled_objs(sampled_bop_objs, transparent=False):
    for j, obj in enumerate(sampled_bop_objs):
        obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.set_shading_mode('auto')
        set_obj_material(obj, transparent=transparent)
        obj.set_cp("category_id", 1)
        # obj.hide() # not working, must have objects in scene within blenderproc
        obj.set_location([1000, 1000, 1000]) # workaround, move it away

def float_sampled_objs(sampled_bop_objs, transparent=False):
    for j, obj in enumerate(sampled_bop_objs):
        obj.enable_rigidbody(False, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.set_shading_mode('auto')
        set_obj_material(obj, transparent=transparent)
        obj.set_cp("category_id", 1)
        x_x = np.random.uniform(-0.15, 0.15)
        y_y = np.random.uniform(-0.15, 0.15)
        z_z = np.random.uniform(0, 0.2)
        # obj.hide() # not working, must have objects in scene within blenderproc
        obj.set_location([x_x, y_y, z_z]) 
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

def random_lights(max_num=4):
    import bpy
    def random_area_light():
        # Create a new area light
        light_data = bpy.data.lights.new(name="Area Light", type='AREA')
        light_object = bpy.data.objects.new(name="Area Light", object_data=light_data)

        # Link the light object to the scene
        bpy.context.collection.objects.link(light_object)
        # Set the shape of the area light to rectangle
        light_data.shape = 'RECTANGLE'
        # Set the size in the x and y directions
        light_data.size = np.random.uniform(0.3, 1)  # Size in the x direction
        light_data.size_y = np.random.uniform(0.3, 1)  # Size in the y direction
        # Set light properties
        light_data.energy = np.random.uniform(50, 100)  # Adjust the energy/intensity of the light
        # light_data.color = (1.0, 1.0, 1.0)  # Set the color of the light (white)
        light_data.color = (np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0))  # Set the color of the light (white)

        delta_dist = 0.5
        # Position the light
        light_object.location = (np.random.uniform(-delta_dist, delta_dist), np.random.uniform(-delta_dist, delta_dist), np.random.uniform(1, 2.5))  # Set the location of the light

        # Optionally, set the rotation of the light (if needed)
        delta_angle = 3.14/12
        light_object.rotation_euler = (np.random.uniform(-delta_angle, delta_angle), np.random.uniform(-delta_angle, delta_angle), np.random.uniform(-delta_angle, delta_angle))  # Adjust the rotation if necessary
            
    def random_3d_position(radius=1, z_min=-1, z_max=1):
        # Random angle between 0 and 2π
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Random z between z_min and z_max
        z = np.random.uniform(z_min, z_max)
        
        # Calculate x and y based on the circular path
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        return (x, y, z)

    def pick_3d_position(radius=1, z_min=-1, z_max=1, theta=0):
        # Random z between z_min and z_max
        z = np.random.uniform(z_min, z_max)
        
        # Calculate x and y based on the circular path
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        return (x, y, z)
            
    def random_point_light(random=False, pick_theta=0):
        # Create a new area light
        light_data = bpy.data.lights.new(name="Point Light", type='POINT')
        light_object = bpy.data.objects.new(name="Point Light", object_data=light_data)

        # Link the light object to the scene
        bpy.context.collection.objects.link(light_object)

        # Set light properties
        light_data.energy = np.random.uniform(100, 300)  # Adjust the energy/intensity of the light
        # light_data.color = (1.0, 1.0, 1.0)  # Set the color of the light (white)
        light_data.color = (np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0))  # Set the color of the light (white)

        # Position the light
        # delta_dist = 1.5
        # light_object.location = (np.random.uniform(-delta_dist, delta_dist), np.random.uniform(-delta_dist, delta_dist), np.random.uniform(1, 2))  # Set the location of the light

        # position around z-axis with radius, point light is used for simulating shadows around box
        radius = np.random.uniform(1, 2)
        if random:
            light_object.location = random_3d_position(radius, 1, 2)
        else:
            light_object.location = pick_3d_position(radius, 1, 2, pick_theta)

    def random_sun_light():
        # Create a new area light
        light_data = bpy.data.lights.new(name="Sun Light", type='SUN')
        light_object = bpy.data.objects.new(name="Sun Light", object_data=light_data)

        # Link the light object to the scene
        bpy.context.collection.objects.link(light_object)

        delta_angle = 3.14/6
        light_data.angle = np.random.uniform(-delta_angle, delta_angle)  # Set the angle of the sun light's shadows
        # Set light properties
        light_data.energy = np.random.uniform(5, 15)  # Adjust the energy/intensity of the light
        light_data.color = (1.0, 1.0, 1.0)  # Set the color of the light (white)

        # Position the light
        delta_dist = 0.5
        light_object.location = (np.random.uniform(-delta_dist, delta_dist), np.random.uniform(-delta_dist, delta_dist), np.random.uniform(1, 2.5))  # Set the location of the light
    
    def random_all_points(max_num=4):
        # for i in range(np.random.randint(2, max_num)):
        #     random_point_light(random=True)

        delta_angle = 3.14/6
        angles = [i * (2*3.14 / max_num) for i in range(max_num)]
        pick_angles = random.sample(angles, 2)
        for pick_angle in pick_angles:
            random_point_light(random=False, pick_theta=np.random.uniform(pick_angle-delta_angle, pick_angle+delta_angle))

        
    def random_all_areas(max_num=4):
        for i in range(np.random.randint(1, max_num)):
            random_area_light()

    def mix_lights(max_num=4):
        for i in range(np.random.randint(1, max_num)):
            light_type = np.random.choice([0,1])
            if light_type == 0:
                random_area_light()
            elif light_type == 1:
                random_point_light()

    light_type = np.random.choice([0,1,2])
    # light_type = np.random.choice([0])
    if light_type == 0:
        random_all_points(max_num)
    elif light_type == 1:
        random_all_areas(max_num)
    else:
        mix_lights(max_num)

def config_gpu_render():
    import bpy
    # Update the scene
    bpy.context.view_layer.update()
    # Get the current scene
    scene = bpy.context.scene
    # Set the exposure time
    exposure_time = np.random.uniform(0.2, 1)  # Adjust this value as needed
    # Check if Cycles render engine is being used
    if scene.render.engine == 'CYCLES':
        scene.cycles.film_exposure = exposure_time
    # Set the device to GPU
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # Use 'CUDA' for NVIDIA, 'OPTIX' for newer NVIDIA cards, or 'HIP' for AMD
    bpy.context.scene.cycles.device = 'GPU'

    # Retrieve and enable all available GPU devices
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Enable all available GPU devices
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = False # disable CPU
        if device.type == 'CUDA' or device.type == 'OPTIX' or device.type == 'HIP':
            device.use = True
            print(f"Enabled device: {device.name}")

def set_camera_pos(baseline):
    def compute_camera_rot(look_vector, up_vector):
        # Normalize the input vectors
        look_vector = look_vector / np.linalg.norm(look_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        # Compute the right vector
        right_vector = np.cross(up_vector, look_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)

        # Recompute the up vector to ensure orthogonality
        up_vector = np.cross(look_vector, right_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        # Create the rotation matrix
        rotation_matrix = np.array([right_vector, up_vector, look_vector]).T
        
        return rotation_matrix

    def random_z_rotation(max_angle=6):
        angle = np.random.randint(0, max_angle)
        gamma = (2*3.14 / max_angle) * angle
        # Define rotation matrices
        R_z = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        R = R_z
        return R
    
    def random_rotation(max_around_x, max_around_y, max_around_z):
        alpha = np.random.uniform(-max_around_x, max_around_x)
        beta = np.random.uniform(-max_around_y, max_around_y)
        gamma = np.random.uniform(-max_around_z, max_around_z)
        # Define rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])

        R_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        R_z = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix (R_z * R_y * R_x)
        R = R_z @ R_y @ R_x
        return R

    # using objects as origin of world coordinate now, later we'll use left camera coordinate as world coordinate
    # Here we use fix distance from scene in z-axis for camera pose

    half_baseline = baseline / 2 
    # cam_z_dist = np.random.uniform(0.8, 1.2) 
    cam_z_dist = np.random.uniform(1.2, 1.3) 
    cam_y_dist = np.random.uniform(-0.1, 0.1) 
    cam_x_dist = np.random.uniform(-0.1, 0.1) 
    # cam_z_dist = 5
    # cam_y_dist = 0
    # cam_x_dist = 0
    alpha = beta = gamma = 0
    if np.random.choice([True, False]):
        alpha = 3.14/12
    if np.random.choice([True, False]):
        beta = 3.14/12
    if np.random.choice([True, False]):
        gamma = 3.14/12
    random_R = random_rotation(alpha, beta, gamma)

    random_zR = random_z_rotation()
    random_R = random_R @ random_zR
    
    cam_left_position = np.array([half_baseline+cam_x_dist, cam_y_dist, cam_z_dist])
    cam_right_position = np.array([-half_baseline+cam_x_dist, cam_y_dist, cam_z_dist])
    cam_left_forward = np.array([half_baseline+cam_x_dist, cam_y_dist, 0]) - cam_left_position
    cam_right_forward = np.array([-half_baseline+cam_x_dist, cam_y_dist, 0]) - cam_right_position
    cam_normalized_left = cam_left_forward / np.sqrt(np.sum(cam_left_forward**2))
    cam_normalized_right = cam_right_forward / np.sqrt(np.sum(cam_right_forward**2))

    cam_left_rotation_matrix = compute_camera_rot(cam_normalized_left, np.array([0,1,0]))
    cam_right_rotation_matrix = compute_camera_rot(cam_normalized_right, np.array([0,1,0]))

    cam_left_rotation_matrix = random_R @ cam_left_rotation_matrix
    cam_right_rotation_matrix = random_R @ cam_right_rotation_matrix
    cam_left_position = random_R @ cam_left_position
    cam_right_position = random_R @ cam_right_position

    cam_left_matrix = bproc.math.build_transformation_mat(cam_left_position, cam_left_rotation_matrix)
    cam_right_matrix = bproc.math.build_transformation_mat(cam_right_position, cam_right_rotation_matrix)
# # Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
    cam_left_matrix = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_left_matrix, ["X", "-Y", "-Z"])
    cam_right_matrix = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_right_matrix, ["X", "-Y", "-Z"])

    bproc.camera.add_camera_pose(cam_left_matrix)
    bproc.camera.add_camera_pose(cam_right_matrix)
    return cam_left_matrix, cam_right_matrix


def compute_obb_from_3d_bbox(bbox_3d, camera_intrinsics, camera_extrinsics):
    """
    Compute the 2D oriented bounding box (OBB) from a 3D bounding box by projecting
    the 3D points to the 2D image plane using the camera's intrinsic and extrinsic matrices.

    Args:
        bbox_3d (list of list of float): The 8 vertices of the 3D bounding box.
        camera_intrinsics (numpy array): The 3x3 camera intrinsic matrix.
        camera_extrinsics (numpy array): The 4x4 camera extrinsic matrix (world to camera).

    Returns:
        box_points (numpy array): 4 corner points of the 2D oriented bounding box.
    """

    # Convert the 3D bounding box points to homogeneous coordinates
    bbox_3d_homogeneous = np.array([list(corner) + [1] for corner in bbox_3d])
    # print(bbox_3d_homogeneous)
    # print(camera_extrinsics)
    # print(camera_intrinsics)

    cam_extri_inverted = np.ones_like(camera_extrinsics)
    cam_extri_inverted[:3,:3] = np.transpose(camera_extrinsics[:3,:3])
    cam_extri_inverted[:3,3] = -np.transpose(camera_extrinsics[:3,:3]) @ camera_extrinsics[:3,3]

    # Project 3D points to 2D image plane using camera intrinsics and extrinsics
    bbox_2d_points = []
    for corner in bbox_3d_homogeneous:
        corner_camera = np.dot(cam_extri_inverted, corner.T)[:3]  # World to camera transformation
        corner_2d = np.dot(camera_intrinsics, corner_camera)
        corner_2d = corner_2d[:2] / corner_2d[2]  # Normalize to get 2D point
        bbox_2d_points.append(corner_2d)

    bbox_2d_points = np.array(bbox_2d_points, dtype=np.float32)
    # print(bbox_2d_points)

    # Compute the minimum-area rotated bounding box in 2D
    rotated_rect = cv2.minAreaRect(bbox_2d_points)
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)

    return box_points

def compute_obb_from_mask(instance_mask):
    """
    Compute the 2D oriented bounding box (OBB) directly from an instance mask.

    Args:
        instance_mask (numpy array): A binary mask where the object is white (1) and the background is black (0).

    Returns:
        box_points (numpy array): 4 corner points of the 2D oriented bounding box.
    """

    # Ensure the mask is in uint8 format
    instance_mask = np.array(instance_mask).astype(np.uint8)

    # Find contours of the object in the mask
    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in the instance mask!")

    # Get the largest contour (assuming single object)
    contour = max(contours, key=cv2.contourArea)

    # Compute the minimum-area oriented bounding box
    rotated_rect = cv2.minAreaRect(contour)

    # Get the four corners of the oriented bounding box
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)

    return box_points

def compute_obb_area(obb_points):
    """
    Compute the area of a 2D oriented bounding box (OBB) given its corner points.

    Args:
        obb_points (numpy array): 4 corner points of the 2D oriented bounding box.

    Returns:
        area (float): Area of the oriented bounding box.
    """
    # Ensure the points are in the correct shape
    assert obb_points.shape == (4, 2), "obb_points must be a 4x2 array."

    # Calculate the width and height of the bounding box
    width = np.linalg.norm(obb_points[0] - obb_points[1])
    height = np.linalg.norm(obb_points[1] - obb_points[2])

    # Compute the area
    area = width * height
    return area

def check_valid_obb(bbox_3d, instance_mask, camera_intrinsics, camera_extrinsics):
    # Example usage:
    # print(bbox_3d)
    # print(instance_mask)
    # print(camera_intrinsics)
    # print(camera_extrinsics)

    # obb_2d_from_3d = compute_obb_from_3d_bbox(bbox_3d, camera_intrinsics, camera_extrinsics)
    # obb_area_3d = compute_obb_area(obb_2d_from_3d)
    # print(obb_2d_from_3d, obb_area_3d)
    # # return True, obb_2d_from_3d

    obb_2d_from_mask = compute_obb_from_mask(instance_mask)
    obb_area_mask = compute_obb_area(obb_2d_from_mask)
    obb_mask_score = np.sum(instance_mask) / obb_area_mask
    # if instance is splited into several parts, we only extract largest part as mask. In this case, instance_mask will be larger than area of mask
    # obb_mask_score = 1.0/obb_mask_score if obb_mask_score > 1 else obb_mask_score

    print(obb_2d_from_mask, obb_area_mask, obb_mask_score)
    # return True, obb_2d_from_mask, obb_mask_score
    if 0.68 < obb_mask_score < 0.96:   # 左边界越大，越认为其是被遮挡(0.68, 1)
        return True, obb_2d_from_mask, obb_mask_score
    else:
        return False, obb_2d_from_mask, obb_mask_score


    # # if the object is occluded, the mask area should be smaller than 3d projection area
    # if obb_area_mask / obb_area_3d > 0.8:
    #     return True, obb_2d_from_3d
    # else:
    #     return False, obb_2d_from_mask

def generate_obb(bbox_3d, instance_mask, camera_intrinsics, camera_extrinsics):
    obb_2d_from_mask = compute_obb_from_mask(instance_mask)
    obb_area_mask = compute_obb_area(obb_2d_from_mask)
    obb_mask_score = np.sum(instance_mask) / obb_area_mask

    return obb_2d_from_mask, obb_mask_score

def extract_obb_from_objs(sampled_objs, render_data, camera_intrics, camera_extrisics):
    def extract_inst_id(obj, instance_map):
        for im in instance_map:
            if im['name'] == obj.get_name():
                return im['idx']
        return -1
    def convert_local_bbox_to_world(obj):
        # Get the object's bounding box in local space
        local_bbox = [mathutils.Vector(corner) for corner in obj.bound_box]

        # Transform the bounding box vertices to world space
        obb = [obj.matrix_world @ vertex for vertex in local_bbox]
        return obb

    # instance_map = render_data['instance_attribute_maps'][0][0]
    # instance_segmaps = render_data['instance_segmaps']
    # print(instance_map)
    # print(instance_segmaps)
    
    obb_attr = [dict(), dict()] 
    for sampled_obj in sampled_objs:
        # extract obb for two views
        instance_map0 = render_data['instance_attribute_maps'][0]
        instance_map1 = render_data['instance_attribute_maps'][1]
        instance_segmap0 = render_data['instance_segmaps'][0]
        instance_segmap1 = render_data['instance_segmaps'][1]

        inst_id0 = extract_inst_id(sampled_obj, instance_map0)
        inst_id1 = extract_inst_id(sampled_obj, instance_map1)

        print(sampled_obj.get_name())
        print(inst_id0, inst_id1)

        if inst_id0 != -1:

            valid0, obb0, score0 = check_valid_obb(convert_local_bbox_to_world(sampled_obj.blender_obj), (instance_segmap0==inst_id0), camera_intrics[0], camera_extrisics[0])
            if valid0:
                obb_attr[0][inst_id0] = { 'obb':obb0, 'score': score0 }
        if inst_id1 != -1:
            valid1, obb1, score1 = check_valid_obb(convert_local_bbox_to_world(sampled_obj.blender_obj), (instance_segmap1==inst_id1), camera_intrics[1], camera_extrisics[1])
            if valid1:
                obb_attr[1][inst_id1] = { 'obb':obb1, 'score': score1 }
    return obb_attr

def extract_obb_from_selected_objs(sampled_objs, render_data, camera_intrics, camera_extrisics):
    def extract_inst_id(obj, instance_map):
        for im in instance_map:
            if im['name'] == obj.get_name():
                return im['idx']
        return -1
    def convert_local_bbox_to_world(obj):
        # Get the object's bounding box in local space
        local_bbox = [mathutils.Vector(corner) for corner in obj.bound_box]

        # Transform the bounding box vertices to world space
        obb = [obj.matrix_world @ vertex for vertex in local_bbox]
        return obb
    
    obb_attr = [dict(), dict()] 
    for j, sampled_obj in enumerate(sampled_objs):
        if j + 1 in selected_list:
            # extract obb for two views
            instance_map0 = render_data['instance_attribute_maps'][0]
            instance_map1 = render_data['instance_attribute_maps'][1]
            instance_segmap0 = render_data['instance_segmaps'][0]
            instance_segmap1 = render_data['instance_segmaps'][1]

            inst_id0 = extract_inst_id(sampled_obj, instance_map0)
            inst_id1 = extract_inst_id(sampled_obj, instance_map1)
 
            print(sampled_obj.get_name())
            print(inst_id0, inst_id1)

            if inst_id0 != -1:
                obb0, score0 = generate_obb(convert_local_bbox_to_world(sampled_obj.blender_obj), (instance_segmap0==inst_id0), camera_intrics[0], camera_extrisics[0])
                obb_attr[0][inst_id0] = { 'obb':obb0, 'score': score0 }
            if inst_id1 != -1:
                obb1, score1 = generate_obb(convert_local_bbox_to_world(sampled_obj.blender_obj), (instance_segmap1==inst_id1), camera_intrics[1], camera_extrisics[1])
                obb_attr[1][inst_id1] = { 'obb':obb1, 'score': score1 }
        else:
            pass
    return obb_attr

def corners_to_rotated_rect(corners):
    """
    Convert four corner points of an oriented bounding box to cv2.RotatedRect.

    Parameters:
    corners (np.array): Array of shape (4, 2) containing the corner points in order.

    Returns:
    cv2.RotatedRect: The corresponding rotated rectangle.
    """
    # Calculate the center
    center_x = (corners[0][0] + corners[2][0]) / 2
    center_y = (corners[0][1] + corners[2][1]) / 2
    center = (center_x, center_y)

    # Calculate width and height
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])

    # Calculate the angle
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle = np.arctan2(dy, dx) * (180 / np.pi)

    return cv2.RotatedRect(center, (width, height), angle) 

def compute_iou(box1, box2):
    # Compute the intersection area
    intersection = cv2.rotatedRectangleIntersection(box1, box2)
    if intersection[0] == 1:
        intersection_area = cv2.contourArea(intersection[1])
    else:
        intersection_area = 0

    # Compute the area of both bounding boxes
    box1_area = box1[1][0] * box1[1][1]  # width * height
    box2_area = box2[1][0] * box2[1][1]  # width * height

    # Compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def nms_obb(boxes, scores, threshold):
    indices = np.argsort(scores)[::-1]  # Sort by scores in descending order
    keep = []

    while len(indices) > 0:
        # Pick the box with the highest score
        current_index = indices[0]
        keep.append(current_index)

        # Compare it with the rest
        current_box = boxes[current_index]
        indices = indices[1:]  # Remove the current box index

        # Filter out boxes with IoU above the threshold
        filtered_indices = []
        for index in indices:
            if compute_iou(current_box, boxes[index]) < threshold:
                filtered_indices.append(index)
        indices = np.array(filtered_indices)

    return keep

# # Example usage
# boxes = [((50, 50), (20, 40), 30), ((55, 55), (20, 40), 30), ((200, 200), (40, 20), 45)]
# scores = [0.9, 0.75, 0.8]
# threshold = 0.3

# selected_indices = nms_obb(boxes, scores, threshold)
# selected_boxes = [boxes[i] for i in selected_indices]

# print("Selected boxes:", selected_boxes)

def nms_for_obb(obbs, threshold=0.2):
    boxes = []
    origin_boxes = []
    scores = []
    inst_indices = []
    for idx, value in obbs.items():
        origin_boxes.append(value['obb'])
        boxes.append(cv2.minAreaRect(value['obb']))
        scores.append(value['score'])
        inst_indices.append(idx)
    
    new_obb = dict() 
    selected_indices = nms_obb(boxes, scores, threshold)
    for i in selected_indices:
        new_obb[inst_indices[i]] = {"obb": origin_boxes[i], 'score': scores[i]}
    return new_obb

def area_select_for_obb(obbs, threshold = 0.75):
    origin_boxes = []
    areas = []
    scores = []
    inst_indices = []
    for idx, value in obbs.items():
        origin_boxes.append(value['obb'])
        areas.append(compute_obb_area(value['obb']))
        scores.append(value['score'])
        inst_indices.append(idx)
    if not areas:
        return {}

    max_area = max(areas)

    new_obb = dict()
    for idx, area in zip(inst_indices, areas):
        if area / max_area > threshold:
            new_obb[idx] = {'obb':origin_boxes[inst_indices.index(idx)], 'score':scores[inst_indices.index(idx)]}
    return new_obb

def match_obbs_in_stereo_image(left_obbs, right_obbs, left_depth, right_depth):
    matched_obbs = []
    return matched_obbs

def crop_obb(image, obb_corners):
    # Convert corners to the right type
    obb_corners = np.array(obb_corners, dtype=np.int32)

    # Create a mask with the same dimensions as the image, filled with the background color (black)
    mask = np.zeros_like(image)

    # Fill the OBB area in the mask with white
    cv2.fillPoly(mask, [obb_corners], (255, 255, 255))

    # Create the output image by combining the mask with the original image
    output_image = np.where(mask == 255, image, 0)

    return output_image

def scale_obb(obb_corners, scale_factor=[1,1], center_bias=[0,0]):
     # Calculate the center of the OBB
    center = np.mean(obb_corners, axis=0)

    # Scale the corners
    centrelized_corners = (obb_corners - center)
    centrelized_corners[:,0] *= scale_factor[0]
    centrelized_corners[:,1] *= scale_factor[1]

    scaled_corners = centrelized_corners + center + center_bias
    return scaled_corners

def merge_obb_to_aabb(obb1, obb2):
    # obb1 and obb2 are 2D numpy arrays with shape (4, 2)
    
    # Stack the corners of both OBBs
    all_corners = np.vstack((obb1, obb2))
    
    # Find the minimum and maximum x and y coordinates
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    # Create the AABB as a numpy array
    aabb = np.array([[min_x, min_y], [max_x, max_y]])
    
    return aabb

def update_image(image, target_size, is_disp=False):
        def padding_image(image, target_ratio):
            """
            Pads an image to the specified aspect ratio.

            :param image: PIL.Image object
            :param target_ratio: desired aspect ratio (width / height)
            :return: new PIL.Image object with the desired aspect ratio
            """
            # Get current image dimensions
            width, height = image.size
            current_ratio = width / height

            if current_ratio > target_ratio:
                # Image is wider than target ratio, pad top and bottom
                new_height = width / target_ratio
                padding_vertical = int((new_height - height) / 2)
                padding = (0, padding_vertical, 0, padding_vertical)
            else:
                # Image is taller than target ratio, pad left and right
                new_width = height * target_ratio
                padding_horizontal = int((new_width - width) / 2)
                padding = (padding_horizontal, 0, padding_horizontal, 0)

            # Create a new image with the desired aspect ratio and the same mode as the original image
            new_image = ImageOps.expand(image, border=padding, fill='black')

            return new_image, padding
        
        def scale_image(image, target_size, is_disp=False):
            """
            Scales an image to the target size.

            :param image: PIL.Image object
            :param target_size: tuple (width, height)
            :return: new PIL.Image object with the target size
            """

            if not is_disp:
                return image.resize(target_size, Image.Resampling.LANCZOS)
            else:
                width, height = image.size
                scale_value = target_size[0] / width
                scaled_image = image.resize(target_size, Image.Resampling.LANCZOS) * scale_value
                return scaled_image

        new_image, padding = padding_image(image, target_ratio=target_size[0]/target_size[1])
        new_image = scale_image(new_image, target_size, is_disp=is_disp)
        return new_image

# online generation for training
def crop_stereo_gt_with_matched_obbs(matched_obb, left_img, right_img, left_disparity, target_size=[640,480]):
    left_obb = matched_obb[0]
    right_obb = matched_obb[1]

    # scale obb for augmentation
    left_obb = scale_obb(left_obb, scale_factor=[np.random.uniform(0.9, 1.2), np.random.uniform(0.9, 1.2)], center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    right_obb = scale_obb(right_obb, scale_factor=[np.random.uniform(0.9, 1.2), np.random.uniform(0.9, 1.2)], center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])

    # crop original image with aug_obb, the croped size is the same as original size
    croped_left = crop_obb(left_img, left_obb)
    croped_right = crop_obb(right_img, right_obb)
    croped_disp = crop_obb(left_disparity, left_obb)

    # merge left and right obb into an aabb for new image
    merged_aabb = merge_obb_to_aabb(left_obb, right_obb)
    # crop again with aabb
    croped_left = crop_obb(croped_left, merged_aabb)
    croped_right = crop_obb(croped_right, merged_aabb)
    croped_disp = crop_obb(croped_disp, merged_aabb)

    # padding and scale with target size
    new_left_image = update_image(croped_left, target_size, is_disp=False)
    new_right_image = update_image(croped_right, target_size, is_disp=False)
    new_left_disp = update_image(croped_disp, target_size, is_disp=True)
    return new_left_image, new_right_image, new_left_disp

    # for idx, obb in left_obbs.items():
    #     left_obb = obb['obb']
    #     right_obb = right_obbs[idx]['obb']
    #     croped_left = crop_obb(left_img, left_obb)
    #     croped_right = crop_obb(right_img, right_obb)

    pass


def generate_rendering():
    parser = argparse.ArgumentParser()
    parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
    parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
    parser.add_argument('bop_dataset_mm', nargs='?', help="Whether unit of BOP dataset model is mm or not")
    parser.add_argument('cc_textures_path', nargs='?', default="resources/cc0_textures", help="Path to downloaded cc textures")
    parser.add_argument('coco_textures_path', nargs='?', default="resources/coco", help="Path to downloaded cc textures")
    parser.add_argument('obj_id', nargs='?', default="1", help="Object id for physics positioning")
    parser.add_argument('obj_num', nargs='?', default="1", help="Number of objects for physics positioning")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
    parser.add_argument('background', nargs='?', default="False", help="Generating background data")
    parser.add_argument('box_width', nargs='?', default=0.5, help="Box width of background", type=float)
    args = parser.parse_args()

    # Set the background color
    bg_c = sample_gray()
    bproc.init(horizon_color=[bg_c, bg_c, bg_c])

    config_gpu_render()

    # four types of generated images
    # 0. no background
    # 1. with bg + box
    # 2. with bg + objs + box
    # 3. with bg + primitives
    # 4. with bg + objs + box + primitive
    img_types = [0, 2, 4]
    # img_types = [0]

    # class id 
    # 0: background
    # 1: object
    # 2: box
    # unoccupied = False

    # load a random sample of bop objects into the scene
    sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                                    mm2m = True if args.bop_dataset_mm == "True" else False,
                                    obj_ids=[int(args.obj_id)] * int(args.obj_num)
                                    )
    for i, obj in enumerate(sampled_bop_objs, start = 1):
        obj.set_cp('obj_id', i)  # set instance id as index
    # load BOP datset intrinsics
    K, width, height, baseline = bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))
    
    # load coco images as bg 
    bg_images = load_images(args.coco_textures_path)
    # load cc pbr textures
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

    # add lights
    random_lights(max_num=4)
    
    # add stereo camera
    cam_left, cam_right = set_camera_pos(baseline)

    transparent = True
    cur_img_type = np.random.choice(img_types)
    cur_img_type = 2
    if cur_img_type == 0:
        # add objs with box, no background
        x_w, y_w, w_delta = add_random_objs_with_box(sampled_bop_objs, bg_images, cc_textures, transparent=transparent)
        set_obj_poses(sampled_bop_objs, x_w, y_w, w_delta)
    elif cur_img_type == 1:
        # hide objs
        hide_sampled_objs(sampled_bop_objs, transparent=transparent)
        # add bg plane
        add_random_bg_planes(bg_images, max_num=1)
        # add box
        add_random_box(bg_images, cc_textures)
    elif cur_img_type == 2:
        # add objs with box
        x_w, y_w, w_delta = add_random_objs_with_box(sampled_bop_objs, bg_images, cc_textures, transparent=transparent)
        set_obj_poses(sampled_bop_objs, x_w, y_w, w_delta)
        # add bg plane
        add_random_bg_planes(bg_images, max_num=1)
    elif cur_img_type == 3:
        # hide objs
        hide_sampled_objs(sampled_bop_objs, transparent=transparent)
        # add bg plane
        add_random_bg_planes(bg_images, max_num=1)
        # add primitives
        add_random_obj(bg_images, max_num=30)
    elif cur_img_type == 4:
        # add objs with box
        x_w, y_w, w_delta = add_random_objs_with_box(sampled_bop_objs, bg_images, cc_textures, transparent=transparent)
        set_obj_poses(sampled_bop_objs, x_w, y_w, w_delta)
        # add bg plane
        add_random_bg_planes(bg_images, max_num=1)
        # add primitives
        add_random_obj(bg_images, x_bound=x_w, y_bound=y_w, max_num=30)
    elif cur_img_type == 5:
        # add obj，have ont box
        add_random_objs_without_box(sampled_bop_objs, bg_images, cc_textures, transparent=transparent)
        # float obj to certain location
        # float_sampled_objs(sampled_bop_objs, transparent=True)
        # hide objs
        hide_sampled_objs(sampled_bop_objs, transparent=transparent)
        # add bg plane
        add_random_bg_planes(bg_images, max_num=1)


    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1,
                                                    max_simulation_time=5,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(512)

    # render the whole pipeline
    data = bproc.renderer.render()

    # # colors are rendered as rgb format
    # left_gray = cv2.cvtColor(np.array(data["colors"][0]), cv2.COLOR_RGB2GRAY)
    # right_gray = cv2.cvtColor(np.array(data["colors"][1]), cv2.COLOR_RGB2GRAY)
    # gray_images = [left_gray, right_gray]

    # Write data in bop format
    if not unoccupied:
        bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                            dataset = args.bop_dataset_name,
                            depths = data["depth"],
                            colors = data["colors"], 
                            # colors = gray_images, 
                            # color_file_format = "JPEG",
                            ignore_dist_thres = 10)
    
    # Render segmentation masks (per class and per instance) 
    # bug exists: https://github.com/DLR-RM/BlenderProc/issues/692
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    # add obb data
    obbs = extract_obb_from_objs(sampled_bop_objs, data, [K, K], [cam_left, cam_right])
    new_obbs0 = area_select_for_obb(obbs[0])
    new_obbs1 = area_select_for_obb(obbs[1])
    obbs = [new_obbs0, new_obbs1]
    new_obbs0 = nms_for_obb(obbs[0], threshold=0.3)
    new_obbs1 = nms_for_obb(obbs[1], threshold=0.3)
    obbs = [new_obbs0, new_obbs1]
    # print(obbs)
    data.update({'obbs':obbs})

    selected_objs = {}
    global selected_list
    inst_id_to_obj = {obj.get_cp('obj_id'): obj for obj in sampled_bop_objs}  # 创建字典，保存所有物体的id和obj
    # for inst_id, obj in inst_id_to_obj.items():
    #     print(f"obj_id: {inst_id}, Object: {obj}")
    # for inst_id, obb_info in obbs[0].items():
    #     print(f"inst_id:{inst_id}")
    selected_objs = {key: inst_id_to_obj[key] for key in inst_id_to_obj if key in obbs[0]}
    # print("selected_list:")
    # print(selected_list)
    # print("\nselected_objs dictionary:")
    # for key, value in selected_objs.items():
    #     print(f"key: {key}, obj: {value}")
        # print(f"Object Name: {value.get_name()}")
    selected_list = list(selected_objs.keys())

    image = cv2.cvtColor(np.array(data['colors'][0]), cv2.COLOR_BGR2RGB)
    obb2draw=[]
    for obb in obbs[0].values():
        obb2draw.append(obb['obb'])
        
    # Draw the bounding box on the image
    cv2.polylines(image, obb2draw, isClosed=True, color=(0, 255, 0), thickness=1)

    # Save the image to a file
    cv2.imwrite('obb_image0.png', image)

    image = cv2.cvtColor(np.array(data['colors'][1]), cv2.COLOR_BGR2RGB)
    obb2draw=[]
    for obb in obbs[1].values():
        obb2draw.append(obb['obb'])

    # Draw the bounding box on the image
    cv2.polylines(image, obb2draw, isClosed=True, color=(0, 255, 0), thickness=1)

    # Save the image to a file
    cv2.imwrite('obb_image1.png', image)



    # # Render segmentation masks (per class and per instance)
    # # bug exists: https://github.com/DLR-RM/BlenderProc/issues/692
    # ret = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])

    # # print(ret)
    # data.update(ret)

    # import matplotlib.pyplot as plt
    # from matplotlib import colors
    # # Define a colormap with three colors
    # cmap = colors.ListedColormap(['red', 'green', 'blue'])

    # # Define the color boundaries
    # bounds = [-0.5, 0.5, 1.5, 2.5]
    # norm = colors.BoundaryNorm(bounds, cmap.N)

    # # Plot the array
    # plt.imshow(data["class_segmaps"][0], cmap=cmap, norm=norm)
    # plt.colorbar()  # To show the color scale

    # # Save the image to a file
    # plt.savefig('array_visualization.png')  # You can change the file name and format


    # write the data to a .hdf5 container
    if not unoccupied:
        # update unoccuiped objs's mask in below two lines, pfm remain the same as origin
        update_objs(sampled_bop_objs, bg_images, cc_textures, transparent=True)

        # 在筛选出来的物体上进行obb选取，保证左右物体一致
        obbs = extract_obb_from_selected_objs(sampled_bop_objs, data, [K, K], [cam_left, cam_right])
        data.update({'obbs':obbs})

        # 可视化
        image = cv2.cvtColor(np.array(data['colors'][0]), cv2.COLOR_BGR2RGB)
        obb2draw=[]
        for obb in obbs[0].values():
            obb2draw.append(obb['obb'])
        cv2.polylines(image, obb2draw, isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.imwrite('obb_image2.png', image)
        image = cv2.cvtColor(np.array(data['colors'][1]), cv2.COLOR_BGR2RGB)
        obb2draw=[]
        for obb in obbs[1].values():
            obb2draw.append(obb['obb'])
        cv2.polylines(image, obb2draw, isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.imwrite('obb_image3.png', image)

        data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))
        bproc.writer.write_hdf5(os.path.join(args.output_dir, 'bop_data', args.bop_dataset_name, 'train_pbr', '000000'), data)


    if unoccupied:
        # remain unoccupied objs, remove other objs
        update_objs(sampled_bop_objs, bg_images, cc_textures, transparent=True)
        # # add random background
        # add_random_bg_planes(bg_images, max_num=1)

        # bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1,
        #                                                 max_simulation_time=5,
        #                                                 check_object_interval=1,
        #                                                 substeps_per_frame = 20,
        #                                                 solver_iters=25)
        # bproc.renderer.set_max_amount_of_samples(512)
        data = bproc.renderer.render()
        bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data2'),
                            dataset = args.bop_dataset_name,
                            depths = data["depth"],
                            colors = data["colors"], 
                            ignore_dist_thres = 10)
          
        data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))
        bproc.writer.write_hdf5(os.path.join(args.output_dir, 'bop_data2', args.bop_dataset_name, 'train_pbr', '000000'), data)


def show_h5():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_path', nargs='?', help="Path to the bop datasets parent directory")
    args = parser.parse_args()

    label = read_hdf5(args.h5_path)

def show_stereo_h5():
    parser = argparse.ArgumentParser()
    parser.add_argument('left_h5_path', nargs='?', help="Path to the bop datasets parent directory")
    parser.add_argument('right_h5_path', nargs='?', help="Path to the bop datasets parent directory")
    args = parser.parse_args()

    read_stereo_hdf5(args.left_h5_path, args.right_h5_path)

# show_h5()
# show_stereo_h5()

# root_path = "./"
# sim_id = "101"
# scene_id = "1"
# rectify_stereo_scene(root_path, sim_id, scene_id, "test")
selected_list = []
generate_rendering()

# for sid in range(1):
#     sim_id = "101"
#     scene_id = str(sid+1)
#     # rectify_stereo_scene("./", sim_id, scene_id, "val", "psm-sim101-val")
#     rectify_stereo_scene("./", sim_id, scene_id, "train", "test")