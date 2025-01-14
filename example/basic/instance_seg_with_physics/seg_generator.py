import blenderproc as bproc
import argparse
import os
import random
import numpy as np
import h5py

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

def compare_mask_ratio(single_mask, mixed_mask, label_idx):
    count_s = np.count_nonzero(single_mask == label_idx)
    count_m = np.count_nonzero(mixed_mask == label_idx)
    return count_m/count_s

def compare_mask_ratio_approximation(object_num, mixed_mask, label_idx):
    ## zero in mixed_mask means null value in depth map
    count_m = np.count_nonzero(mixed_mask == label_idx)
    full_mask = np.copy(mixed_mask)
    full_mask = full_mask[(np.where((full_mask <= object_num) & (full_mask > 0)))] # don't consider background, remove null values
    max_count = np.max(np.bincount(full_mask))
    return count_m/max_count

def save_complete_object_mask_with_scores(object_labels, total_label, output_file_path):
    retain_id = []
    retain_scores = {}
    retain_rescores = {}
    for obj_idx, object_label in enumerate(object_labels):
        retain_score = compare_mask_ratio(object_label, total_label, obj_idx+1)
        retain_id.append(obj_idx+1)
        retain_scores[(obj_idx+1)] = retain_score

    bg_id = []
    bg_scores = {}
    bg_rescores = {}
    total_obj_num = np.max(total_label)
    for bg_idx in range(len(object_labels)+1, total_obj_num+1):
        bg_id.append(bg_idx)
        bg_scores[bg_idx] = 0.0

    print("retain_ids:", retain_id)
    print("bg_ids:", bg_id)

    new_label = total_label.copy().astype(np.int32)

    # here we reorganize the labels:
    # "complete": 1 -> #complete_object
    # "background": 1000 + 1 -> 1000 + #background_object

    # complete
    for idx, rid in enumerate(retain_id):
        new_label[np.where(total_label==rid)] = idx+1  # label start from 1
        retain_rescores[str(idx+1)] = retain_scores[rid]

    # background
    for idx, rid in enumerate(bg_id):
        new_label[np.where(total_label==rid)] = ((idx+1) + 1000)
        bg_rescores[str((idx+1) + 1000)] = bg_scores[rid]

    print("label:", np.unique(new_label))

    new_scores = retain_rescores.copy()
    new_scores.update(bg_rescores)
    print("new_scores:", new_scores)
    
    #write_hdf5(output_file_path, new_label)
    write_hdf5_with_scores(output_file_path, new_label, new_scores)

def save_object_score_with_approximation(object_labels, total_label, output_file_path):
    retain_id = []
    retain_scores = {}
    retain_rescores = {}

    for obj_idx, object_label in enumerate(object_labels):
        retain_score = compare_mask_ratio_approximation(len(object_labels), total_label, obj_idx+1)
        retain_id.append(obj_idx+1)
        retain_scores[(obj_idx+1)] = retain_score

    bg_id = []
    bg_scores = {}
    bg_rescores = {}
    total_obj_num = np.max(total_label)
    for bg_idx in range(len(object_labels)+1, total_obj_num+1):
        bg_id.append(bg_idx)
        bg_scores[bg_idx] = 0.0

    print("retain_ids:", retain_id)
    print("bg_ids:", bg_id)

    new_label = total_label.copy().astype(np.int32)

    # here we reorganize the labels:
    # "complete": 1 -> #complete_object
    # "background": 1000 + 1 -> 1000 + #background_object

    # complete
    for idx, rid in enumerate(retain_id):
        new_label[np.where(total_label==rid)] = idx+1  # label start from 1
        retain_rescores[str(idx+1)] = retain_scores[rid]

    # background
    for idx, rid in enumerate(bg_id):
        new_label[np.where(total_label==rid)] = ((idx+1) + 1000)
        bg_rescores[str((idx+1) + 1000)] = bg_scores[rid]

    print("label:", np.unique(new_label))

    new_scores = retain_rescores.copy()
    new_scores.update(bg_rescores)
    print("new_scores:", new_scores)
    
    #write_hdf5(output_file_path, new_label)
    write_hdf5_with_scores(output_file_path, new_label, new_scores)

def save_complete_object_mask(object_labels, total_label, output_file_path, ratio=0.95):
    retain_id = []
    remove_id = []
    bg_id = []
    for obj_idx, object_label in enumerate(object_labels):
        if compare_mask_ratio(object_label, total_label, obj_idx+1) > ratio:
            retain_id.append(obj_idx+1)
        else:
            remove_id.append(obj_idx+1)

    total_obj_num = np.max(total_label)
    for bg_idx in range(len(object_labels)+1, total_obj_num+1):
        bg_id.append(bg_idx)
    
    print("retain_ids:", retain_id)
    print("remove_ids:", remove_id)
    print("bg_ids:", bg_id)

    new_label = total_label.copy().astype(np.int32)

   # for idx, rid in enumerate(retain_id):
   #     new_label[np.where(total_label==rid)] = idx+1  # label start from 1

   # for idx, rid in enumerate(remove_id):
   #     new_label[np.where(total_label==rid)] = len(retain_id)+1 # background

    # here we reorganize the labels:
    # "complete": 1 -> #complete_object
    # "non-complete": 1000 + 1 -> 1000 + #complete_object
    # "background": 2000 + 1 -> 2000 + #background_object

    # complete
    for idx, rid in enumerate(retain_id):
        new_label[np.where(total_label==rid)] = idx+1  # label start from 1

    # non-complete
    for idx, rid in enumerate(remove_id):
        new_label[np.where(total_label==rid)] = ((idx+1) + 1000)

    # background
    for idx, rid in enumerate(bg_id):
        new_label[np.where(total_label==rid)] = ((idx+1) + 2000)

    print("label:", np.unique(new_label))
    
    write_hdf5(output_file_path, new_label)
    

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

def merge_labels(obj_num, h5_folder, h5_mixed_file, output_h5_file):
    single_labels = []
    for render_idx in range(obj_num):
        cur_h5_file = os.path.join(h5_folder, str(render_idx), '0.hdf5')
        #if os.path.exists(cur_h5_file):
        cur_label = read_hdf5(cur_h5_file)
        single_labels.append(cur_label)
        
    mixed_label = read_hdf5(h5_mixed_file)
    #save_complete_object_mask(single_labels, mixed_label, output_h5_file, ratio=0.7)
    save_complete_object_mask_with_scores(single_labels, mixed_label, output_h5_file)

def merge_labels_v2(obj_num, h5_mixed_file, output_h5_file):
   # single_labels = []
   # for render_idx in range(obj_num):
   #     cur_h5_file = os.path.join(h5_folder, str(render_idx), '0.hdf5')
   #     #if os.path.exists(cur_h5_file):
   #     cur_label = read_hdf5(cur_h5_file)
   #     single_labels.append(cur_label)
   #     
    mixed_label = read_hdf5(h5_mixed_file)
    #save_complete_object_mask(single_labels, mixed_label, output_h5_file, ratio=0.7)
    save_object_score_with_approximation(np.arange(0, obj_num), mixed_label, output_h5_file)


def generate_rendering():
    parser = argparse.ArgumentParser()
    parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
    parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
    parser.add_argument('bop_dataset_mm', nargs='?', help="Whether unit of BOP dataset model is mm or not")
    parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
    parser.add_argument('obj_id', nargs='?', default="1", help="Object id for physics positioning")
    parser.add_argument('obj_num', nargs='?', default="1", help="Number of objects for physics positioning")
    parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
    parser.add_argument('background', nargs='?', default="False", help="Generating background data")
    parser.add_argument('box_width', nargs='?', default=0.5, help="Box width of background", type=float)
    args = parser.parse_args()

    bproc.init()

    # load a random sample of bop objects into the scene
    sampled_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                                    mm2m = True if args.bop_dataset_mm == "True" else False,
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
            #mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
            mat.set_principled_shader_value("Base Color", [1,0,0, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            
    # create room
    x_l = 0.5
    y_l = 0.5
    z_l = 0.05 # z-axis scale not working in blenderproc!
   # x_w = 0.5
   # y_w = 0.5
    x_w = args.box_width
    y_w = args.box_width
    w_delta = -0.1
    room_planes = [bproc.object.create_primitive('PLANE', scale=[x_w, y_w, 1]),
                bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, -y_w, 0], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_w, y_l, z_l], location=[0, y_w, 0], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[x_w, 0, 0], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('CUBE', scale=[x_l, y_w, z_l], location=[-x_w, 0, 0], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    neg_planes = []
    if args.background == "True":
        # use randomly special plane for adding background data
        for i in range(0, random.randint(1, 5)):
            b_x = random.uniform(0.02, 0.04)
            b_y = random.uniform(0.02, y_w/2)
            b_z = random.uniform(0.02, 0.04)
            bg = bproc.object.create_primitive('CUBE', scale=[b_x, b_y, b_z])
            bg.enable_rigidbody(False, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            bg.set_location([random.uniform(-x_w/2, x_w/2),random.uniform(-y_w/2, y_w/2),random.uniform(0.05, 0.5)])
            bg.set_rotation_euler([random.uniform(0, 3.1415/6),random.uniform(0, 3.1415/6),random.uniform(0, 3.1415)])
            neg_planes.append(bg)

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

    import math
    #euler_angles=[0,0,0]
    angle_xs = [0, math.pi]
    angle_ys = [0, math.pi]

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
        obj.set_location(pos)
        #obj.set_rotation_euler(bproc.sampler.uniformSO3())
        obj.set_rotation_euler(sample_xyz_pose(angle_xs, angle_ys))

    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)

    ## Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1,
                                                    max_simulation_time=5,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop_objs)


    # Here we use fix distance from scene in z-axis for camera pose
    cam_position = [0, 0, 1.5]
    cam_rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, -1])
    cam_matrix = bproc.math.build_transformation_mat(cam_position, cam_rotation_matrix)
    bproc.camera.add_camera_pose(cam_matrix)

    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(1)

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
    merge_labels(int(args.obj_num), render_mask, h5_mixed_file, h5_filtered_file)


    # approximation for score without rendering
 #   merge_labels_v2(int(args.obj_num), h5_mixed_file, h5_filtered_file)


def show_h5():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_path', nargs='?', help="Path to the bop datasets parent directory")
    args = parser.parse_args()

    label = read_hdf5(args.h5_path)


#show_h5()
generate_rendering()