import json
import argparse
import re
import cv2
import open3d as o3d
import numpy as np
import h5py
import chardet 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def convert_depth_to_points(intri, depth_img, depth_trunc=10000):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intri['width'], intri['height'], intri['fx'], intri['fy'], intri['cx'], intri['cy'])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img, intrinsic=intrinsic, extrinsic=np.identity(4),
        depth_scale=1000, depth_trunc=depth_trunc, stride=int(1))
    
    
   # print(np.max(depth_img))
   # print(np.min(depth_img))
   # print(pcd.points)
    return pcd

def read_camera_intrinsic(instrinsic_json):
    with open(instrinsic_json, 'r') as f:
        scene_camera = json.load(f)
    #print(intrinsic)
    return scene_camera

def read_raw_depth(depth_img_path):
    depth_raw = o3d.io.read_image(depth_img_path)
    np_depth = np.array(depth_raw)
   # print(np.max(np_depth))
   # print(np.min(np_depth))
   # print(np_depth)
    depth_raw = o3d.geometry.Image(np_depth)
    return depth_raw


def read_hdf5(hdf5_path):
    filename = hdf5_path
    with h5py.File(filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        left_depth = np.array(f["depth"][:])
       # print(np.max(label_img))
       # print(np.min(label_img))
    depth_raw = o3d.geometry.Image(left_depth)

    return depth_raw

def show_points():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_intrinsic', nargs='?', help="Path to the camera instrinsic json")
    parser.add_argument('depth_img', nargs='?', help="Path to the depth image")
    args = parser.parse_args()

    depth_raw = read_hdf5(args.depth_img)
    intrinsic = read_camera_intrinsic(args.camera_intrinsic)
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("origin_points.ply", pcd)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_pfm_wo_flip(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception(f"文件头错误，找到: {header}，应为 'PF' 或 'Pf'。")

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

def show_disparity():
    parser = argparse.ArgumentParser()
    parser.add_argument('left_image', nargs='?', help="Path to the left image")
    parser.add_argument('right_image', nargs='?', help="Path to the right image")
    parser.add_argument('disparity_map', nargs='?', help="Path to the disparity image")
    args = parser.parse_args()

    
 # Read the images
    import cv2
    left_image = cv2.imread(args.left_image, cv2.IMREAD_COLOR)
    right_image = cv2.imread(args.right_image, cv2.IMREAD_COLOR)
    disparity_map, scale = read_pfm_wo_flip(args.disparity_map)
    # disparity_map, scale = readPFM(args.disparity_map)
    disparity_map = np.ascontiguousarray(disparity_map,dtype=np.float32)
    # disparity_map = (disparity_map * scale).astype(np.float32)
    # disparity_map = (disparity_map * scale)
    print(disparity_map)
    print(scale)

    left_line, right_line = draw_horizontal_lines(left_image, right_image)

    pixel = [160, 297]
    for i in range(-10, 10):
        print("continous disp: ", disparity_map[pixel[1] + i, pixel[0]])
    for i in range(-10, 10):
        depth = 400 * 0.4 / disparity_map[pixel[1] + i, pixel[0]]
        print("depth: ", depth)

    # Draw circles on the left image
    cv2.circle(left_line, pixel, 4, (255, 0, 0), 2)

    # Calculate the corresponding coordinates on the right image
    corresponding_coords = (round(pixel[0] - disparity_map[pixel[1], pixel[0]]), pixel[1])
    # corresponding_coords = (round(pixel[0] - 17.87), pixel[1])
    print(disparity_map.shape)
    print(corresponding_coords)
    print(disparity_map[pixel[1], pixel[0]], pixel[1])

    # Draw circles on the right image
    cv2.circle(right_line, corresponding_coords, 4, (0, 255, 0), 2)

    import matplotlib.pyplot as plt

     # Create a figure with subplots

     # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Display the left image
    # axes[0].imshow(left_image)
    axes[0].imshow(left_line)
    axes[0].set_title('Left Image')
    axes[0].axis('off')

    # Display the right image
    # axes[1].imshow(right_image)
    axes[1].imshow(right_line)
    axes[1].set_title('Right Image')
    axes[1].axis('off')

    plt.savefig("mydisparity.png")


def draw_horizontal_lines(img1, img2, num_lines=10):
    """
    Draw horizontal lines across stereo images.

    Args:
    img1: First stereo image (left image).
    img2: Second stereo image (right image).
    num_lines: Number of horizontal lines to draw.

    Returns:
    Tuple of images with horizontal lines drawn.
    """
    # Copy images to avoid modifying the originals
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    # Get image dimensions
    h, w, _ = img1.shape

    # Calculate the spacing between lines
    spacing = h // (num_lines + 1)

    # Draw horizontal lines
    for i in range(1, num_lines + 1):
        y = i * spacing
        cv2.line(img1_copy, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(img2_copy, (0, y), (w, y), (0, 255, 0), 1)
    
    return img1_copy, img2_copy

def disparity_to_point_cloud_wo_Q(disparity_map, focal_length, baseline, cx, cy):
    Q = np.identity(4)
    Q[2][2] = 0.0
    Q[0][3] = -cx
    Q[1][3] = -cy
    Q[2][3] = focal_length 
    Q[3][2] =  1.0/baseline
    Q[3][3] =  0.0
    print(Q)
    return disparity_to_point_cloud(disparity_map, Q)

def disparity_to_point_cloud_with_mask(disparity_map, Q, mask_pixels):
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
    roi_mask = generate_mask(disparity_map, mask_pixels)

    mask = roi_mask & mask
    points_3D = points_3D[mask]
    
    return points_3D, mask

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
    
    return points_3D, mask

def show_rectified_colored_points(colors, points, filename="recover_from_disparity_wt_color.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename, pcd)

def show_rectified_points(points, filename="recover_from_disparity.ply"):
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
def show_points_from_disparity_Q():
    parser = argparse.ArgumentParser()
    parser.add_argument('left_disparity', nargs='?', help="Path to the disparity image")
    parser.add_argument('Q', nargs='?', help="Path to the Q file")
    args = parser.parse_args()

    # Read the JSON file and convert it to a list
    with open(args.Q, 'r') as file:
        Q_file = json.load(file)

    # Convert the list to a NumPy array
    Q = np.array(Q_file)

    left_disparity, _ = read_pfm_wo_flip(args.left_disparity)
    print(left_disparity)
    print(left_disparity.shape)

    # show_rectified_points(disparity_to_point_cloud(left_disparity, Q))
    show_rectified_points(disparity_to_point_cloud(left_disparity, Q))

def generate_mask(image, mask_pixels):
    # Create a blank mask image
    mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw the mask based on the provided pixels
    draw.polygon(mask_pixels, fill=255)
    mask_np = np.array(mask)
    # Threshold the image to create a boolean mask
    boolean_mask = mask_np > 0
    return boolean_mask

def apply_mask(image, mask_pixels, outside_color=[70,70,70]):
   # Check if the image is RGB or Grayscale
    if image.mode == 'RGB':
        image_np = np.array(image)
        is_rgb = True
    elif image.mode == 'L':
        image_np = np.array(image)
        is_rgb = False
    elif image.mode == 'F':
        image = image.convert('F')
        image_np = np.array(image)
        is_rgb = False
    else:
        raise ValueError("Unsupported image mode. Only RGB and Grayscale (L or F) images are supported.")
    
    # Create a blank mask image
    mask = Image.new("L", (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw the mask based on the provided pixels
    draw.polygon(mask_pixels, fill=255)
    mask_np = np.array(mask)
    
    if is_rgb:
        # Create an RGB version of the mask with the outside color
        masked_image_np = np.where(mask_np[:, :, None] == 255, image_np, outside_color)
        masked_image = Image.fromarray(masked_image_np.astype('uint8'), 'RGB')
    else:
        # Apply mask for grayscale image
        outside_value = outside_color[0] if isinstance(outside_color, (list, tuple)) else outside_color
        if image.mode == 'F':
            masked_image_np = np.where(mask_np == 255, image_np, outside_value)
            masked_image = Image.fromarray(masked_image_np.astype('float32'), 'F')
        else:
            masked_image_np = np.where(mask_np == 255, image_np, outside_value)
            masked_image = Image.fromarray(masked_image_np.astype('uint8'), 'L')
    
    return masked_image


def show_colored_points_from_disparity_Q(roi=False, mask_pixels=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('left_disparity', nargs='?', help="Path to the disparity image")
    parser.add_argument('left_rgb', nargs='?', help="Path to the color image")
    parser.add_argument('Q', nargs='?', help="Path to the Q file")
    args = parser.parse_args()

    # Read the JSON file and convert it to a list
    with open(args.Q, 'r') as file:
        Q_file = json.load(file)

    # Convert the list to a NumPy array
    Q = np.array(Q_file)

    left_disparity, _ = read_pfm_wo_flip(args.left_disparity)
    left_image = cv2.imread(args.left_rgb, cv2.IMREAD_COLOR)

    colors = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0
    print(left_disparity)
    print(left_disparity.shape)
    if roi:
        points, mask = disparity_to_point_cloud_with_mask(left_disparity, Q, mask_pixels)
    else:
        points, mask = disparity_to_point_cloud(left_disparity, Q)
    colors = colors[mask]

    # show_rectified_points(disparity_to_point_cloud(left_disparity, Q))
    show_rectified_colored_points(colors, points)

def visualize_scene_flow_disparity_error():
    """
    Visualize the error between the ground truth and predicted disparity maps.

    Parameters:
    gt_disparity (numpy.ndarray): Ground truth disparity map.
    pred_disparity (numpy.ndarray): Predicted disparity map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_disparity', nargs='?', help="Path to the disparity image")
    parser.add_argument('pred_disparity', nargs='?', help="Path to the disparity image")
    args = parser.parse_args()

    # Compute the absolute error
    gt_disparity, _ = readPFM(args.gt_disparity)
    pred_disparity, _ = read_pfm_wo_flip(args.pred_disparity)
    error_map = np.abs(gt_disparity - pred_disparity[4:,:]) # have 4 pixel padding on top when testing (544, 960)
    avg_error = np.mean(error_map)

    print("gt disparity map: ", np.min(gt_disparity), np.max(gt_disparity))
    print("pred disparity map: ", np.min(pred_disparity), np.max(pred_disparity))
    print("error map: ", np.min(error_map), np.max(error_map), np.mean(error_map))

    # Normalize the error map for better visualization
    error_map_normalized = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))
    show_rectified_points(disparity_to_point_cloud_wo_Q(gt_disparity, 1050, 1, 479.5, 269.5), "gt_depth.ply") # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    show_rectified_points(disparity_to_point_cloud_wo_Q(pred_disparity, 1050, 1, 479.5, 269.5), "pred_depth.ply") # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    # Plot the error map
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Error')
    plt.title('Disparity Error Visualization')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()


def visualize_disparity_error():
    """
    Visualize the error between the ground truth and predicted disparity maps.

    Parameters:
    gt_disparity (numpy.ndarray): Ground truth disparity map.
    pred_disparity (numpy.ndarray): Predicted disparity map.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_disparity', nargs='?', help="Path to the disparity image")
    parser.add_argument('pred_disparity', nargs='?', help="Path to the disparity image")
    args = parser.parse_args()

    # Compute the absolute error
    gt_disparity, _ = read_pfm_wo_flip(args.gt_disparity)
    pred_disparity, _ = read_pfm_wo_flip(args.pred_disparity)
    error_map = np.abs(gt_disparity - pred_disparity)
    avg_error = np.mean(error_map)

    print("error map: ", np.min(error_map), np.max(error_map), np.mean(error_map))

    # Normalize the error map for better visualization
    error_map_normalized = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))

    # Plot the error map
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Error')
    plt.title('Disparity Error Visualization')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

def convert_to_grayscale():
    """
    Convert an image to grayscale and save it.
    
    Parameters:
    input_image_path (str): The path to the input image.
    output_image_path (str): The path to save the grayscale image.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path', nargs='?', help="Path to the disparity image")
    parser.add_argument('output_image_path', nargs='?', help="Path to the color image")
    args = parser.parse_args()
    # Read the image
    img = cv2.imread(args.input_image_path)
    
    # Check if the image was successfully read
    if img is None:
        print(f"Error: Unable to read image {args.input_image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(grayscale_image.shape)

    grayscale_image = grayscale_image[..., ::-1].copy()
    print(grayscale_image.shape)
    
    # Save the grayscale image
    cv2.imwrite(args.output_image_path, grayscale_image)
    print(f"Grayscale image saved to {args.output_image_path}")


def sgm_disp():
    parser = argparse.ArgumentParser()
    parser.add_argument('left_rgb', nargs='?', help="Path to the disparity image")
    parser.add_argument('right_rgb', nargs='?', help="Path to the color image")
    args = parser.parse_args()

    rectified_left_rgb = cv2.imread(args.left_rgb, cv2.IMREAD_COLOR)
    rectified_right_rgb = cv2.imread(args.right_rgb, cv2.IMREAD_COLOR)
    # Set parameters for the SGM algorithm
    window_size = 5
    min_disp = 0
    num_disp = 256  # Should be divisible by 16
    # Convert rectified images to grayscale
    gray_left = cv2.cvtColor(rectified_left_rgb, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right_rgb, cv2.COLOR_BGR2GRAY)

    # Compute the disparity map
    stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
    )
    disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16
    save_pfm("test_disparity.pfm", disparity_map)

    disparity_map = (disparity_map*256).astype('uint16')
    disparity_map = Image.fromarray(disparity_map)
    disparity_map.save('Test_disparity.png')

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

mask_pixels = [(262, 152), (446, 152), (446, 352), (262, 352)] 
# show_points()
# show_disparity()
# show_points_from_disparity_Q()
show_colored_points_from_disparity_Q(roi=False, mask_pixels=mask_pixels)
# visualize_disparity_error()
# visualize_scene_flow_disparity_error()
# convert_to_grayscale()
# sgm_disp()