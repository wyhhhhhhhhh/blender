import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import open3d as o3d
import cv2
import copy
import argparse
import h5py


def render_to_file(h5_file, save_file):
    # Create a renderer with the desired image size
    img_width = 640
    img_height = 480
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # Pick a background colour (default is light gray)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA

    print(h5_file)
    with h5py.File(h5_file, 'r') as hf:
        points = hf['points'][:]
        colors = hf['colors'][:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Define a simple unlit Material.
    # (The base color does not replace the arrows' own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # Add the arrow mesh to the scene.
    # (These are thicker than the main axis arrows, but the same length.)
    #render.scene.add_geometry("rotated_model", mesh_r, mtl)
    render.scene.add_geometry("instance_seg", pcd, mtl)

    # Since the arrow material is unlit, it is not necessary to change the scene lighting.
    #render.scene.scene.enable_sun_light(False)
    #render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 15.0  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    center = [0, 0, 0]  # look_at target
    eye = [0, 0, 5]  # camera position
    up = [0, 1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)

    # Read the image into a variable
    img_o3d = render.render_to_image()

    # Display the image in a separate window
    # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
    img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Preview window", img_cv2)
    cv2.waitKey()

    # Optionally write it to a PNG file
    o3d.io.write_image(save_file, img_o3d, 9)


def capture_render(h5_file, save_file):
    with h5py.File(h5_file, 'r') as hf:
        points = hf['points'][:]
        colors = hf['colors'][:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=640)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(-30.0)
    ctr.set_lookat((0, 0, 0))
    ctr.set_front((0, 0, -1))
    ctr.set_up((0, 1, 0))

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_file)
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_folder', nargs='?', help="Path to the point clouds")
    parser.add_argument('save_folder', nargs='?', help="Path to the rendering image")
    args = parser.parse_args()

    if not os.path.exists(args.h5_folder):
        exit(0)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    h5_files = os.listdir(args.h5_folder)
    for h5_file in h5_files:
        h5_file_path = os.path.join(args.h5_folder, h5_file)
        filename, file_extension =os.path.splitext(os.path.basename(h5_file))
        if file_extension == '.h5':
            render_image_path = os.path.join(args.save_folder, filename + ".png")
            #render_to_file(h5_file_path, render_image_path)
            capture_render(h5_file_path, render_image_path)