import open3d as o3d
import numpy as np
import argparse
import os

def voxel_downsample_pcd(input_file, output_file, scale_size=1, voxel_size=0.01):
    pcd = o3d.io.read_point_cloud(input_file)
    pcd_scale = np.asarray(pcd.points) * scale_size
    pcd.points = o3d.utility.Vector3dVector(pcd_scale)

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(output_file, voxel_down_pcd)

def uniform_downsample_pcd(input_file, output_file,k = 10):
    pcd = o3d.io.read_point_cloud(input_file)
    uniform_down_pcd = pcd.uniform_down_sample(k)
    o3d.io.write_point_cloud(output_file, uniform_down_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', nargs='?', help="Path to the point clouds")
    parser.add_argument('output_folder', nargs='?', help="Path to the outputs")
    parser.add_argument('scale_size', nargs='?', help="scale size")
    parser.add_argument('voxel_size', nargs='?', help="voxel size")
    args = parser.parse_args()

    print(args.input_folder)
    print(args.output_folder)
    if not os.path.exists(args.input_folder):
        exit(0)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    ply_files = os.listdir(args.input_folder)
    for ply_file in ply_files:
        ply_file_path = os.path.join(args.input_folder, ply_file)
        filename, file_extension =os.path.splitext(os.path.basename(ply_file))
        if file_extension == '.ply':
            output_path = os.path.join(args.output_folder, filename + "_vh_clean_2.ply")
            voxel_downsample_pcd(ply_file_path, output_path, float(args.scale_size), float(args.voxel_size))
            #uniform_downsample_pcd(ply_file_path, output_path, args.voxel_size)