import blenderproc as bproc
import argparse
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('object', nargs='?', default="obj_000004.ply", help="Path to the model file")
parser.add_argument('output_dir', nargs='?', default="output", help="Path to where the final files will be saved")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj(args.object)[0]
# Use vertex color for texturing
for mat in obj.get_materials():
    mat.map_vertex_color()
# Set pose of object via local-to-world transformation matrix
# obj.set_local2world_mat(
#     [[0.331458, -0.9415833, 0.05963787, -0.04474526765165741],
#     [-0.6064861, -0.2610635, -0.7510136, 0.08970402424862098],
#     [0.7227108, 0.2127592, -0.6575879, 0.6823395750305427],
#     [0, 0, 0, 1.0]]
# )
obj.set_local2world_mat(
    [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1.0]]
)
# Scale 3D model from mm to m
obj.set_scale([0.001, 0.001, 0.001])
# Set category id which will be used in the BopWriter
obj.set_cp("category_id", 1)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

def compute_camera_rot(look_vector, up_vector):
    # Normalize the input vectors
    look_vector = look_vector / np.linalg.norm(look_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # Compute the right vector
    right_vector = np.cross(up_vector, look_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # Recompute the up vector to ensure orthogonality
    up_vector = np.cross(look_vector, right_vector)

    # Create the rotation matrix
    rotation_matrix = np.array([right_vector, up_vector, look_vector]).T
    
    return rotation_matrix


# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[537.4799, 0.0, 318.8965],
     [0.0, 536.1447, 238.3781],
     [0.0, 0.0, 1.0]], 640, 480
)

poi = bproc.object.compute_poi([obj])
# poi
print(poi)

location = np.array([0,0,-0.2])
# location = location / np.linalg.norm(location)
look = poi - location
look = look / np.linalg.norm(look)
print(look)
 # Compute rotation based on vector going from location towards poi
# rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
rotation_matrix = compute_camera_rot(look, [0,1,0])
# Add homog cam pose based on location an rotation
cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
print(cam2world_matrix)
cam2world_matrix = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world_matrix, ["X", "-Y", "-Z"])
print(cam2world_matrix)

# # Set camera pose via cam-to-world transformation matrix
# cam2world = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, -0.5],
#     [0, 0, 0, 1]
# ])
# # Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
# cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
# print(cam2world)
# print(cam2world_matrix)
bproc.camera.add_camera_pose(cam2world_matrix)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()

# Write object poses, color and depth in bop format
bproc.writer.write_bop(args.output_dir, [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)
