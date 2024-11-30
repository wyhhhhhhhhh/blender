from __future__ import print_function
import numpy as np
import time
import json
import cv2
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from plyfile import PlyData, PlyElement


from models import *

import sys
import os
# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the current script's directory to the Python path
sys.path.append(current_script_directory)


model_file = os.path.join(current_script_directory, "checkpoint_150.tar")
# model_file = os.path.join(current_script_directory, "checkpoint_155.tar")
# model_file = os.path.join(current_script_directory, "checkpoint_260.tar")
Q_file = os.path.join(current_script_directory,"Q.json")

with open(Q_file, 'r') as file:
    Q = np.array(json.load(file))


normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

maxdisp=256

seed = 1

roi = True
mask_pixels = [(267, 98), (443, 98), (443, 221), (269, 222)] 

def load_model():
    cuda = torch.cuda.is_available()
    assert cuda

        # model, optimizer
    model = PSMNet(
                                    maxdisp=maxdisp,
                                    ndisps=[64, 32],
                                    disp_interval_pixel=[4, 1],
                                    cr_base_chs=[32, 32],
                                    grad_method="detach",
                                    using_ns=True,
                                    ns_size=3)


    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    # load parameters
    print("loading model {}".format(model_file))
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    return model

def disparity_to_depth(disparity_map, focal_length, baseline):
    """
    Convert a disparity map to a depth map.

    Parameters:
    disparity_map (numpy.ndarray): Input disparity map.
    focal_length (float): Focal length of the camera.
    baseline (float): Baseline distance between the two cameras.

    Returns:
    numpy.ndarray: Output depth map.
    """
    # Avoid division by zero by setting very small disparities to a minimum value
    disparity_map[disparity_map == 0] = 0.1

    # Calculate depth map
    depth_map = (focal_length * baseline) / disparity_map

    return depth_map

def save_disparity_png(disp, filename="disparity.png"):
    img = disp
    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save(filename)

def passive_stereo(left_image, right_image):
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    # left_image = Image.open(left_image).convert('RGB')
    # right_image = Image.open(right_image).convert('RGB')
    colors = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0
    # print(colors)
    # print(colors.shape)
    # print(type(colors))

    left_image = Image.fromarray(left_image).convert('RGB')
    right_image = Image.fromarray(right_image).convert('RGB')
    left_image.save("rgb_left.png")
    right_image.save("rgb_right.png")


    imgL = infer_transform(left_image)
    imgR = infer_transform(right_image) 
    
    # I will do padding in capturing step 
    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = inference(imgL,imgR)
    pred_depth = disparity_to_depth(pred_disp, Q[2][3], -1.0/Q[3][2])
    if roi:
        points, mask = disparity_to_point_cloud_with_mask(pred_disp, Q, mask_pixels)
    else:
        points, mask = disparity_to_point_cloud(pred_disp, Q)
    colors = colors[mask]
    print('time = %.2f' %(time.time() - start_time))
    save_pfm(os.path.join(current_script_directory, "disparity.pfm"), pred_disp)
    save_disparity_png(pred_disp, os.path.join(current_script_directory, "disparity.png"))
    save_depth_map_as_png(pred_depth, os.path.join(current_script_directory,"depth.png"))
    # save_point_cloud(points, os.path.join(current_script_directory,"points.ply"))
    save_rectified_colored_points(colors, points, os.path.join(current_script_directory,"points.ply"))
    return points

def save_rectified_colored_points(colors, points, filename="recover_from_disparity_wt_color.ply"):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename, pcd)

def inference(imgL,imgR):
    model = load_model()
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        outputs = model(imgL,imgR)
        outputs_stage = outputs["stage{}".format(2)]
        disp = outputs_stage["pred"][-1]
        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

    return pred_disp


def save_depth_map_as_png(depth_map, filename):
    """
    Save the depth map as a PNG file.

    Parameters:
    depth_map (numpy.ndarray): Input depth map.
    filename (str): Filename to save the depth map.
    """
    # Normalize the depth map for better visualization
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Convert to colormap
    colormap = cm.plasma(normalized_depth_map)

    # Save as PNG
    plt.imsave(filename, colormap)


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
    Convert a disparity map to a point cloud using the Q matrix.

    Parameters:
    disparity_map (numpy.ndarray): Input disparity map.
    Q (numpy.ndarray): The 4x4 reprojection matrix.

    Returns:
    numpy.ndarray: Array of points in the point cloud.
    """
    # Reproject image to 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    
    # Get the mask of points with valid disparity values
    mask = disparity_map > disparity_map.min()
    
    # Extract valid points
    points = points_3D[mask]
    
    return points, mask

def save_point_cloud(points, filename):
    """
    Save the point cloud to a PLY file.

    Parameters:
    points (numpy.ndarray): Array of points in the point cloud.
    filename (str): Filename to save the point cloud.
    """
    vertices = np.array(
        [(point[0], point[1], point[2]) for point in points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    ply_element = PlyElement.describe(vertices, 'vertex')
    PlyData([ply_element]).write(filename)


if __name__ == "__main__":
    left_image = Image.open("stereo2D_left.png").convert("RGB")
    right_image = Image.open("stereo2D_right.png").convert("RGB")
    # left_image = Image.open("0002_left.png").convert("RGB")
    # right_image = Image.open("0002_right.png").convert("RGB")

    left_image = np.array(left_image)
    right_image = np.array(right_image)

    passive_stereo(left_image, right_image)
    # passive_stereo("stereo2D_left.png", "stereo2D_right.png")