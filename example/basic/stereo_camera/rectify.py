# import blenderproc as bproc
import argparse
import json
import os
import random
import shutil
import numpy as np
import h5py
import cv2
from PIL import Image
from PIL import Image, ImageOps
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colors

def show_points(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename, pcd)


    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # #vis.get_render_option().background_color = [0,0,0]
    # vis.run()
    # vis.destroy_window()

def read_stereo_hdf5(left_hdf5_path, right_hdf5_path, scene_camera_path):
    left_filename = left_hdf5_path
    right_filename = right_hdf5_path
    with h5py.File(left_filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        left_image = np.array(f["colors"][:])
        left_depth = np.array(f["depth"][:])
        left_class_mask = np.array(f["class_segmaps"][:])
        # print(left_image.shape)
        # print(left_depth.shape)
        # print(left_class_mask.shape)
        # print("left depth:", np.min(left_depth))
        # print("left depth:", np.max(left_depth))
        # print("left mask:", np.min(left_class_mask))
        # print("left mask:", np.max(left_class_mask))
        left_class_ins = np.array(f["instance_segmaps"][:])
        # print(np.min(ins), np.max(ins))
    with h5py.File(right_filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        right_image = np.array(f["colors"][:])
        right_depth = np.array(f["depth"][:])
        right_class_mask = np.array(f["class_segmaps"][:])
        # print("right mask:", np.min(right_class_mask))
        # print("right mask:", np.max(right_class_mask))
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
        # T = inv_right_T @ left_T
        R = T[:3,:3]
        t = T[:3, 3]
    return left_image, right_image, left_depth, right_depth, K, R, t, left_class_mask, right_class_mask, left_class_ins

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
    # mask = disparity_map > disparity_map.min()
    mask = np.logical_and(disparity_map > 10, disparity_map < 255)
    points_3D = points_3D[mask]
    print(np.min(disparity_map), np.max(disparity_map))
    print(np.min(points_3D), np.max(points_3D))
    
    return points_3D

def depth_to_disparity(depth, Q):
    # 获取 Q 矩阵的逆矩阵
    Q_inv = np.linalg.inv(Q)
    
    # 创建视差图
    height, width = depth.shape[:2]
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # 遍历每个像素，计算对应的视差值
    for y in range(height):
        for x in range(width):
            # 获取深度值
            Z = depth[y, x]
            
            # 根据 Q 矩阵计算视差值
            if Z != 0:  # 确保深度不为零以避免除零错误
                X = x
                Y = y
                W = 1.0
                
                # 计算归一化坐标
                vec = np.array([X, Y, Z, W])
                vec = np.dot(Q_inv, vec)
                vec /= vec[3]  # 归一化
                
                # 获取视差值
                disparity[y, x] = vec[2]  # vec[2] 是视差值
            else:
                disparity[y, x] = 0  # 如果深度为零，则视差也为零
    
    return disparity

def link_images_to_dotav1(source_image_path, dotav1_path, data_type="train"):
    output_image_path = os.path.join(dotav1_path, "images", data_type)
    for filename in os.listdir(source_image_path):
        source_file = os.path.join(source_image_path, filename)
        target_file = os.path.join(output_image_path, filename)
        if os.path.isfile(source_file):  # Check if it's a file
            shutil.copy2(source_file, target_file)


def convert_obb_to_dotav1(hdf5_path, image_name, dotav1_path, data_type="train"):
    obb_list = []
    with h5py.File(hdf5_path, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        obbs = json.loads(f["obbs"][()].decode('utf-8'))
        # print(obbs)
        for inst_idx, obb in obbs.items():
            corners = np.array(obb['obb']).flatten().tolist()
            obb_list.append(corners)
    
    output_label_file = os.path.join(dotav1_path, "labels", data_type, image_name + ".txt")
    with open(output_label_file, 'w') as f:
        for bbox in obb_list:
            line = ' '.join(map(str, bbox))  # Convert each bbox to a space-separated string
            line = line + " plane 0" # use dota image class label
            f.write(line + '\n')  # Write each bbox on a new line
        f.close()

def rectify_stereo_scene(root_path, sim_id, scene_id, mode, output_path):
    cur_path = os.path.join(root_path, "sim"+sim_id, mode, "Scene_{}".format(scene_id), "bop_data", "lm", "train_pbr", "000000")
    left_h5 = os.path.join(cur_path, "0.hdf5")
    right_h5 = os.path.join(cur_path, "1.hdf5")
    scene_camera = os.path.join(cur_path, "scene_camera.json")

    if not os.path.exists(left_h5):
        return 
    left_image, right_image, left_depth, right_depth, K, R, t, left_class_mask, right_class_mask, left_class_ins = read_stereo_hdf5(left_h5, right_h5, scene_camera)
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

    # mask for object, box and background
    simid_class_mask_folder = os.path.join(output_path, "sim"+sim_id+"_class_mask")
    scene_folder = os.path.join(simid_class_mask_folder, "scene")
    class_mask_left = os.path.join(scene_folder, "left")
    class_mask_right = os.path.join(scene_folder, "right")
    if not os.path.exists(class_mask_left):
        os.makedirs(class_mask_left, exist_ok=True)
    if not os.path.exists(class_mask_right):
        os.makedirs(class_mask_right, exist_ok=True)

    # ins for object
    simid_class_ins_folder = os.path.join(output_path, "sim"+sim_id+"_class_ins")
    scene_folder = os.path.join(simid_class_ins_folder, "scene")
    class_ins_left = os.path.join(scene_folder, "left")
    class_ins_right = os.path.join(scene_folder, "right")
    if not os.path.exists(class_ins_left):
        os.makedirs(class_ins_left, exist_ok=True)
    if not os.path.exists(class_ins_right):
        os.makedirs(class_ins_right, exist_ok=True)
    
    # local patch of objects
    simid_local_patch_folder = os.path.join(output_path, "sim"+sim_id+"_local_object")
    scene_folder = os.path.join(simid_local_patch_folder, "scene")
    local_patch_left = os.path.join(scene_folder, "left")
    local_patch_right = os.path.join(scene_folder, "right")
    if not os.path.exists(local_patch_left):
        os.makedirs(local_patch_left, exist_ok=True)
    if not os.path.exists(local_patch_right):
        os.makedirs(local_patch_right, exist_ok=True)

    
    # convert to gray 
    left_gray = cv2.cvtColor(rectified_left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(rectified_right_rgb, cv2.COLOR_RGB2GRAY)
    left_image = Image.fromarray(left_gray)
    right_image = Image.fromarray(right_gray)

    # left_image = Image.fromarray(rectified_left_rgb)
    # right_image = Image.fromarray(rectified_right_rgb)
    
    left_image.save(os.path.join(cleanpass_left, f'{int(scene_id):04d}.png'))
    right_image.save(os.path.join(cleanpass_right, f'{int(scene_id):04d}.png'))

    save_pfm(os.path.join(disparity_left, f'{int(scene_id):04d}.pfm'), left_disparity)

    np.save(os.path.join(class_mask_left, f'{int(scene_id):04d}.npy'), left_class_mask)
    np.save(os.path.join(class_mask_right, f'{int(scene_id):04d}.npy'), right_class_mask)

    np.save(os.path.join(class_ins_left, f'{int(scene_id):04d}.npy'), left_class_ins)

    # Define a colormap with three colors
    cmap = colors.ListedColormap(['red', 'green', 'blue'])
    # Define the color boundaries
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Plot the array
    plt.imshow(left_class_mask, cmap=cmap, norm=norm)
    plt.colorbar()  # To show the color scale
    # Save the image to a file
    plt.savefig(os.path.join(class_mask_left, f'{int(scene_id):04d}.png'))  # You can change the file name and format
    plt.close()

    plt.imshow(right_class_mask, cmap=cmap, norm=norm)
    plt.colorbar()  # To show the color scale
    plt.savefig(os.path.join(class_mask_right, f'{int(scene_id):04d}.png'))  # You can change the file name and format
    plt.close()

    plt.imshow(left_class_ins)
    plt.colorbar()  # To show the color scale
    plt.savefig(os.path.join(class_ins_left, f'{int(scene_id):04d}.png'))  # You can change the file name and format
    plt.close()

    with open("Q.json", 'w') as file:
        import json
        json.dump(Q.tolist(), file)
    # show_points(disparity_to_point_cloud(left_disparity, Q))

    # save local info of obbs
    with h5py.File(left_h5, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        left_obbs = json.loads(f["obbs"][()].decode('utf-8'))
        local_patch_left_json = os.path.join(local_patch_left, f'{int(scene_id):04d}.json') 
        with open(local_patch_left_json, 'w') as file:
            json.dump(left_obbs, file)

    with h5py.File(right_h5, "r") as f:
        right_obbs = json.loads(f["obbs"][()].decode('utf-8'))
        local_patch_right_json = os.path.join(local_patch_right, f'{int(scene_id):04d}.json') 
        with open(local_patch_right_json, 'w') as file:
            json.dump(right_obbs, file)

    # find common object keys of left and right images
    common_keys = left_obbs.keys() & right_obbs.keys()
    for idx in common_keys:
        left_obb = left_obbs[idx]['obb']
        right_obb = right_obbs[idx]['obb']
        print(left_obb, right_obb)
        # new_left, new_right, new_disp = crop_stereo_gt_with_matched_obbs([left_obb, right_obb], left_image, right_image, left_disparity, target_size=[640, 480])
        # new_left, new_right, new_disp = crop_stereo_gt_with_offset_matched_obbs([left_obb, right_obb], left_image, right_image, left_disparity, target_size=[640, 480])
        # new_left, new_right, new_disp = crop_stereo_gt_with_offset_matched_obbs([left_obb, right_obb], left_image, right_image, left_disparity)

def convert_all_to_dotav1(origin_path, psm_path, sim_id, scene_length, mode, output_path):
    images = os.path.join(output_path, "images", mode)
    labels = os.path.join(output_path, "labels", mode)
    if not os.path.exists(images):
        os.makedirs(images, exist_ok=True)
    if not os.path.exists(labels):
        os.makedirs(labels, exist_ok=True)
    # link left images as training images
    simid_cleanpass_folder = os.path.join(psm_path, "sim"+sim_id+"_frames_cleanpass")
    scene_folder = os.path.join(simid_cleanpass_folder, "scene")
    cleanpass_left = os.path.join(scene_folder, "left")
    link_images_to_dotav1(cleanpass_left, output_path, mode)

    # convert obb label of each image
    for sid in range(scene_length):
        scene_id = str(sid+1)
        cur_path = os.path.join(origin_path, "sim"+sim_id, mode, "Scene_{}".format(scene_id), "bop_data", "lm", "train_pbr", "000000")
        left_h5 = os.path.join(cur_path, "0.hdf5")
        image_name = str("{:04}".format(int(scene_id)))
        convert_obb_to_dotav1(left_h5, image_name, output_path, mode)

# we only use left images as training image for oriented object detection
def convert_to_dotav1(root_path, sim_id, scene_id, mode, output_path):
    cur_path = os.path.join(root_path, "sim"+sim_id, mode, "Scene_{}".format(scene_id), "bop_data", "lm", "train_pbr", "000000")
    left_h5 = os.path.join(cur_path, "0.hdf5")
    right_h5 = os.path.join(cur_path, "1.hdf5")

    # save all to output
    simid_cleanpass_folder = os.path.join(output_path, "sim"+sim_id+"_frames_cleanpass")
    scene_folder = os.path.join(simid_cleanpass_folder, "scene")
    cleanpass_left = os.path.join(scene_folder, "left")
    cleanpass_right = os.path.join(scene_folder, "right")
    link_images_to_dotav1(cleanpass_left)

def disparity_to_depth(disparity_map, focal_length, baseline):
    """
    Convert a disparity map to a depth map.

    Parameters:
    disparity_map (numpy.ndarray): Input disparity map.
    focal_length (float): Focal length of the camera.
    baseline (float): Baseline distance between the two cameras.

    Returns:
    numpy.ndarray: Depth map.
    """
    # Avoid division by zero by setting disparity to a small value where it is zero
    disparity_map = np.where(disparity_map == 0, 100000, disparity_map)

    # Calculate depth map
    depth_map = (focal_length * baseline) / disparity_map

    return depth_map

def write_depth_to_h5(depth_data, h5_filename, dataset_name='depth'):
    """
    Write a depth map to an H5 file.

    Parameters:
    depth_data (numpy.ndarray): The depth map data to be written.
    h5_filename (str): The name of the H5 file to write to.
    dataset_name (str): The name of the dataset in the H5 file.
    """
    with h5py.File(h5_filename, 'w') as h5_file:
        h5_file.create_dataset(dataset_name, data=depth_data)
    print(f"Depth data successfully written to {h5_filename} under dataset name '{dataset_name}'.")

def read_pfm_wo_flip(file):
    import re
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:  # big-endian
            endian = '>'

        # Read the data
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        return np.reshape(data, shape), scale

def compute_disp_by_depth_Q(depth, Q):
    f = Q[2][3]
    q43 = Q[3][2]
    q44 = Q[3][3]
    x_coords, y_coords = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    disparity = (f / depth - q44) / q43 
    return x_coords, y_coords, disparity

def detect_occlusion_vectorized_from_depth(depth_left, depth_right, Q, t=1):
    """
    Detect occlusion using depth maps for the left and right images.
    
    depth_left: Depth map for the left image
    depth_right: Depth map for the right image
    
    Returns:
    - occlusion_mask: Binary mask (1 for occluded regions, 0 otherwise)
    """
    # Convert depth maps to disparity maps
    x_left, y_left, disparity_left = compute_disp_by_depth_Q(depth_left, Q)
    x_right, y_right, disparity_right = compute_disp_by_depth_Q(depth_right, Q)

    height, width = disparity_left.shape

    # # Create a meshgrid of pixel indices
    # x_left = np.tile(np.arange(width), (height, 1))  # x-coordinates for the left image
    # y_left = np.tile(np.arange(height).reshape(-1, 1), (1, width))  # y-coordinates for the left image

    # Compute corresponding x-coordinates in the right image based on disparity
    x_right = np.clip(x_left - np.round(disparity_left).astype(np.int32), 0, width - 1)

    # Gather disparity values from the right disparity map at the computed coordinates
    disparity_right_warped = disparity_right[y_left, x_right]

    # Compute occlusion mask:
    # 1. Pixels where disparity difference is large
    # 2. Invalid disparity (where right image doesn't have a valid match)
    occlusion_mask = np.abs(disparity_left - disparity_right_warped) > t  # Threshold can be adjusted
    return occlusion_mask

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

def scale_obb(obb_corners, scale_factor=1, center_bias=[0,0]):
     # Calculate the center of the OBB
    center = np.mean(obb_corners, axis=0)

    # Scale the corners
    # centrelized_corners = (obb_corners - center)
    # centrelized_corners[:,0] *= scale_factor[0]
    # centrelized_corners[:,1] *= scale_factor[1]

    scaled_corners = (obb_corners - center) * scale_factor + center + center_bias
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
    aabb = np.array([[min_x, min_y], [max_x, max_y]], dtype=np.int32)
    
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
                return np.array(image.resize(target_size, Image.Resampling.LANCZOS))
            else:
                width, height = image.size
                scale_value = target_size[0] / width
                print(scale_value)
                
                scaled_image = cv2.resize(np.array(image), (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST) * scale_value

                # scaled_image = np.array(image.resize(target_size, Image.Resampling.BILINEAR)) * scale_value
                # scaled_image[scaled_image<0] = 0
                return scaled_image

        if not is_disp:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Create a Pillow image from the NumPy array
            pillow_image = Image.fromarray(rgb_image)
            new_image, padding = padding_image(pillow_image, target_ratio=target_size[0]/target_size[1])
            new_image = scale_image(new_image, target_size, is_disp=is_disp)
            # Convert RGB to BGR
            cv2_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        else:
            # Create a Pillow image from the NumPy array
            pillow_image = Image.fromarray(image)
            new_image, padding = padding_image(pillow_image, target_ratio=target_size[0]/target_size[1])
            new_image = scale_image(new_image, target_size, is_disp=is_disp)
            cv2_image = new_image
        return cv2_image


def save_disparity_png(disp, filename="disparity.png"):
    img = disp 
    non_zero_values = img[img!= 0]
    print(np.min(non_zero_values), np.max(non_zero_values))
    img = (img - np.min(non_zero_values)) / (np.max(non_zero_values) - np.min(non_zero_values))
    img[img < 0] = 0
    img = (img*256).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def aabb_length(aabb):
    xmin, ymin = aabb[0,:]
    xmax, ymax = aabb[1,:]
    x_length = xmax - xmin
    y_length = ymax - ymin
    return x_length, y_length

def offset_matched_obbs(left_obb, right_obb, left_img, right_img, left_disp, target_size=[640, 480], target_maxdisp=256):

    # compute new position in target size and maxdisp
    # TODO: if the obb is larger than the whole target size?? very large obb
    def compute_offsets(left_obb, right_obb, target_size, target_maxdisp):
        # compute offset range
        left_aabb = merge_obb_to_aabb(left_obb, left_obb)
        right_aabb = merge_obb_to_aabb(right_obb, right_obb)
        left_x_length, left_y_length = aabb_length(left_aabb)
        right_x_length, right_y_length = aabb_length(right_aabb)
        actual_disp = left_aabb[0,0] - right_aabb[0,0] # use (xmin_l - xmin_r) distance to represent
        actual_y_diff = left_aabb[0,1] - right_aabb[0,1] # ymin

        min_margin = 10
        max_margin = min(target_maxdisp, actual_disp) - 10
        new_disp = np.random.randint(min_margin, max_margin)

        # suppose the right aabb will be pasted on most left of new image
        # determine the new position, left-top position of aabb
        left_right_length = new_disp + left_x_length # length occupied from left aabb to rigth aabb with new disparity
        new_right_pos_x = np.random.randint(0, target_size[0] - left_right_length)
        new_left_pos_x = new_right_pos_x + new_disp
        # we put roi on the mid-height of image ranging from [1/4, 2/4] height
        mid_height = int(target_size[1]/4)
        new_right_pos_y = np.random.randint(mid_height, 2*mid_height)
        new_left_pos_y = new_right_pos_y + actual_y_diff

        new_disparity = new_disp
        offset_disp = actual_disp - new_disp
        return left_aabb, right_aabb, [new_left_pos_x, new_left_pos_y], [new_right_pos_x, new_right_pos_y] , new_disparity, offset_disp
    
    def crop_with_aabb(aabb, img):
        xmin, ymin = aabb[0,:]
        xmax, ymax = aabb[1,:]
        roi = img[ymin:ymax, xmin:xmax]
        return roi
    
    def paste_roi(aabb_roi, target_size, new_pos):
        new_image = np.zeros((target_size[1], target_size[0], 1), dtype=np.uint8)
        new_image[new_pos[1]: new_pos[1] + aabb_roi.shape[0], new_pos[0]: new_pos[0] + aabb_roi.shape[1]] = aabb_roi[:,:,np.newaxis]
        return new_image

    def paste_roi_disp(aabb_roi, target_size, new_pos):
        new_image = np.zeros((target_size[1], target_size[0], 1), dtype=np.float32)
        new_image[new_pos[1]: new_pos[1] + aabb_roi.shape[0], new_pos[0]: new_pos[0] + aabb_roi.shape[1]] = aabb_roi[:,:,np.newaxis]
        return new_image
    
    def crop_with_offsets(left_aabb, right_aabb, left_obb, right_obb, new_left_pos, new_right_pos, offset_disp, left_img, right_img, left_disp, target_size):
        # crop original image with obb, the cropped size is the same as original size, with black background
        cropped_left = crop_obb(left_img, left_obb)
        cropped_right = crop_obb(right_img, right_obb)
        cropped_disp = crop_obb(left_disp, left_obb)

        # use aabb to crop roi
        roi_left = crop_with_aabb(left_aabb, cropped_left)
        roi_right = crop_with_aabb(right_aabb, cropped_right)
        roi_disp = crop_with_aabb(left_aabb, cropped_disp)
        # change original disparity value
        roi_disp = roi_disp - offset_disp
        roi_disp[roi_disp < 0] = 0

        # paste rois into new gray image with new pos
        new_image_left = paste_roi(roi_left, target_size, new_left_pos)
        new_image_right = paste_roi(roi_right, target_size, new_right_pos)
        new_image_disp = paste_roi_disp(roi_disp, target_size, new_left_pos)


        return new_image_left, new_image_right, new_image_disp.squeeze()
    
    left_aabb, right_aabb, new_left_pos, new_right_pos, _, offset_disp = compute_offsets(left_obb, right_obb, target_size, target_maxdisp)
    print("OFFSET")
    print("obb")
    print(left_obb, right_obb)
    print("aabb")
    print(left_aabb, right_aabb)
    print("new pos")
    print(new_left_pos, new_right_pos, offset_disp)
    new_image_left, new_image_right, new_image_disp = crop_with_offsets(left_aabb, right_aabb, left_obb, right_obb, new_left_pos, new_right_pos, offset_disp, left_img, right_img, left_disp, target_size)
    print(new_image_disp.shape, new_image_disp.dtype, np.min(new_image_disp), np.max(new_image_disp))

    cv2.imwrite('new_left.png', new_image_left)
    cv2.imwrite('new_right.png', new_image_right)
    save_disparity_png(new_image_disp, "newdisp.png")
    save_pfm("newdisp.pfm", new_image_disp)
    return new_image_left, new_image_right, new_image_disp

# online generation for training
def crop_stereo_gt_with_offset_matched_obbs(matched_obb, left_img, right_img, left_disparity, target_size=[640,480], target_maxdisp=256):
    left_obb = matched_obb[0]
    right_obb = matched_obb[1]

    # scale obb for augmentation
    left_obb = scale_obb(left_obb, scale_factor=np.random.uniform(0.9, 1.2), center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    right_obb = scale_obb(right_obb, scale_factor=np.random.uniform(0.9, 1.2), center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    print(left_obb, right_obb)

    return offset_matched_obbs(left_obb, right_obb, left_img, right_img, left_disparity, target_size, target_maxdisp)


# online generation for training
def crop_stereo_gt_with_matched_obbs(matched_obb, left_img, right_img, left_disparity, target_size=[640,480]):
    left_obb = matched_obb[0]
    right_obb = matched_obb[1]

    # scale obb for augmentation
    left_obb = scale_obb(left_obb, scale_factor=np.random.uniform(0.9, 1.2), center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    right_obb = scale_obb(right_obb, scale_factor=np.random.uniform(0.9, 1.2), center_bias=[np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
    print(left_obb, right_obb)

    # crop original image with aug_obb, the croped size is the same as original size
    # offset the crop randomly
    # augment the patch with x_offset
    croped_left = crop_obb(left_img, left_obb)
    croped_right = crop_obb(right_img, right_obb)
    croped_disp = crop_obb(left_disparity, left_obb)

     # Save the image to a file
    cv2.imwrite('croped_left.png', croped_left)
    cv2.imwrite('croped_right.png', croped_right)
    save_disparity_png(croped_disp)

    # merge left and right obb into an aabb for new image
    merged_aabb = merge_obb_to_aabb(left_obb, right_obb)
    print(merged_aabb)
    # crop again with aabb
    xmin, ymin = merged_aabb[0,:]
    xmax, ymax = merged_aabb[1,:]
    cropped_left = croped_left[ymin:ymax, xmin:xmax]
    cropped_right = croped_right[ymin:ymax, xmin:xmax]
    cropped_disp = croped_disp[ymin:ymax, xmin:xmax]

    #  # Save the image to a file
    cv2.imwrite('croped_left.png', cropped_left)
    cv2.imwrite('croped_right.png', cropped_right)
    save_disparity_png(cropped_disp)
    # print("FFFF:", cropped_disp.shape, cropped_disp.dtype)

    # padding and scale with target size
    new_left_image = update_image(cropped_left, target_size, is_disp=False)
    new_right_image = update_image(cropped_right, target_size, is_disp=False)
    new_left_disp = update_image(cropped_disp, target_size, is_disp=True)

    #  # Save the image to a file
    cv2.imwrite('new_left.png', new_left_image)
    cv2.imwrite('new_right.png', new_right_image)
    save_disparity_png(new_left_disp, "newdisp.png")
    return new_left_image, new_right_image, new_left_disp

    # for idx, obb in left_obbs.items():
    #     left_obb = obb['obb']
    #     right_obb = right_obbs[idx]['obb']
    #     croped_left = crop_obb(left_img, left_obb)
    #     croped_right = crop_obb(right_img, right_obb)

    pass



def rectify_stereo_images(left_rgb_image, right_rgb_image, left_depth, right_depth, K, R, t):
    # # Load your RGB image and depth map
    # left_rgb_image = cv2.imread(left_image)
    # right_rgb_image = cv2.imread(right_image)
    # left_depth_map = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED)

    distCoeff = np.zeros(5)
    # Rectify RGB image
    print(K)
    # # print(left_rgb_image.shape[:2])
    # print(R)
    # print(t)
    print(-R.T@t)
    print(np.linalg.norm(-R.T@t))
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
    # print(rectified_left_depth)
    # count = np.sum((rectified_left_depth > 0) & (rectified_left_depth < 0.8))
    # print(count)

    # # Set parameters for the SGM algorithm
    # window_size = 5
    # min_disp = 0
    # num_disp = 256  # Should be divisible by 16
    # # Convert rectified images to grayscale
    # gray_left = cv2.cvtColor(rectified_left_rgb, cv2.COLOR_BGR2GRAY)
    # gray_right = cv2.cvtColor(rectified_right_rgb, cv2.COLOR_BGR2GRAY)

    # # Compute the disparity map
    # stereo = cv2.StereoSGBM_create(
    # minDisparity=min_disp,
    # numDisparities=num_disp,
    # blockSize=window_size,
    # P1=8 * 3 * window_size ** 2,
    # P2=32 * 3 * window_size ** 2,
    # disp12MaxDiff=1,
    # uniquenessRatio=10,
    # speckleWindowSize=100,
    # speckleRange=32
    # )
    # disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

    f = Q[2][3]
    q43 = Q[3][2]
    q44 = Q[3][3]
    print(Q)
    # f = 500
    # q43 = 2.5
    # q44 = 0



    left_x_coords, left_y_coords = np.meshgrid(np.arange(rectified_left_depth.shape[1]), np.arange(rectified_left_depth.shape[0]))
    # print(left_x_coords)
    left_disparity = (f / rectified_left_depth - q44) / q43 
    # print("leftdisp: ", left_disparity.shape)
    right_x_coords = left_x_coords - left_disparity
    # print(left_disparity)
    # print(right_y_coords)
    valid_mask = (right_x_coords >= 0)

    # print(valid_mask)
    mask_area = np.logical_and(rectified_left_depth > 0, rectified_left_depth < 10000)
    mask_area = mask_area & valid_mask
    # print(mask_area)

    occ_mask = detect_occlusion_vectorized_from_depth(rectified_left_depth, rectified_right_depth, Q, 2)
    mask_area = mask_area & ~occ_mask

    mask_image = (mask_area * 255).astype(np.uint8)

    # Create a PIL image from the array
    img = Image.fromarray(mask_image)

    # Save the image as a black and white PNG
    img.save('mask_image.png')

    left_disparity = np.zeros_like(rectified_left_depth)
    left_disparity[mask_area] = (f / rectified_left_depth[mask_area] - q44) / q43 
    # print(left_disparity.shape)
    print(np.min(left_disparity), np.max(left_disparity))
    # print(left_disparity)

    # pixel = [1150, 90]
    # print("gt_disp:", left_disparity[pixel[1], pixel[0]])
    # print("sgm_disp:", disparity_map[pixel[1], pixel[0]])
    # print(disparity_map.shape)
    # print(left_disparity[pixel[1], pixel[0]])
    # left_test = np.zeros_like(left_disparity)
    # left_test[pixel[1]][pixel[0]] = left_disparity[pixel[1]][pixel[0]]

    # save_pfm("disp.pfm", left_disparity)
    # left_disparity, _ = read_pfm_wo_flip("disp.pfm")

    # depth_recover = disparity_to_depth(left_test, f, 1.0/q43).astype(np.float32)
    # print(depth_recover.dtype)
    # nK = None
    # # print(distCoeff)
    # inv_map1x, inv_map1y = cv2.initUndistortRectifyMap(P1[:,:3], distCoeff, R1.T, nK, [w, h], cv2.CV_32FC1)
    # unrectified_left_depth = cv2.remap(depth_recover, inv_map1x, inv_map1y, interpolation=cv2.INTER_NEAREST)

    # write_depth_to_h5(unrectified_left_depth, "depth.hdf5")
    # # print(nK)

    # disp = depth_to_disparity(left_disparity, Q)
    # show_points(disparity_to_point_cloud(disparity_map, Q), "recover_from_disparity_sgm.ply")
    # show_points(disparity_to_point_cloud(left_disparity, Q), "recover_from_disparity_gt.ply")
    


    # # Draw circles on the left image
    # cv2.circle(rectified_left_rgb, pixel, 10, (255, 0, 0), 2)
    # corresponding_coords = (round(pixel[0] - disparity_map[pixel[1]][pixel[0]]), pixel[1])
    # cv2.circle(rectified_right_rgb, corresponding_coords, 10, (0, 255, 0), 2)
    # print(left_disparity[pixel[1]][pixel[0]], pixel[1])

    # # Display rectified images
    # from matplotlib import pyplot as plt
    # import matplotlib

    # # # Normalize and display the disparity map
    # # plt.imshow(disparity_map, 'gray')
    # # plt.colorbar()
    # # plt.show()

    # fig, ax = plt.subplots(nrows=4, ncols=2)
    # plt.subplot(4, 2, 1)
    # plt.imshow(left_rgb_image)
    # plt.subplot(4, 2, 2)
    # plt.imshow(right_rgb_image)
    # plt.subplot(4, 2, 3)
    # plt.imshow(rectified_left_rgb)
    # plt.subplot(4, 2, 4)
    # plt.imshow(rectified_right_rgb)
    # plt.subplot(4, 2, 5)
    # plt.imshow(rectified_left_depth)
    # plt.subplot(4, 2, 6)
    # plt.imshow(rectified_right_depth)
    # plt.subplot(4, 2, 7)
    # plt.imshow(left_disparity)
    # plt.subplot(4, 2, 8)
    # plt.imshow(left_disparity)
    # # plt.show(block=False)
    # plt.savefig("mygraph.png")
    # # plt.pause(10)
    # plt.close()

    return rectified_left_rgb, rectified_right_rgb, rectified_left_depth, rectified_right_depth, left_disparity, Q


dataid=3
for sid in range(500):
    sim_id = "{}".format(dataid)
    scene_id = str(sid+1)
    # rectify_stereo_scene("./", sim_id, scene_id, "train", "train".format(dataid))
    rectify_stereo_scene("./", sim_id, scene_id, "train", "psm-sim{}-train".format(dataid))
    # rectify_stereo_scene("./", sim_id, scene_id, "val", "psm-sim110-val")
    # rectify_stereo_scene("./", sim_id, scene_id, "train", "test")# 

for sid in range(20):
    sim_id = "{}".format(dataid)
    scene_id = str(sid+1)
    # rectify_stereo_scene("./", sim_id, scene_id, "train", "psm-sim110-train")
    # rectify_stereo_scene("./", sim_id, scene_id, "val", "val".format(dataid))
    rectify_stereo_scene("./", sim_id, scene_id, "val", "psm-sim{}-val".format(dataid))

# convert_all_to_dotav1("./", "psm-sim{}-train".format(dataid), str(dataid), 500, "train", "sim{}_DotaV1".format(dataid))
# convert_all_to_dotav1("./", "psm-sim{}-val".format(dataid), str(dataid), 40, "val", "sim{}_DotaV1".format(dataid))