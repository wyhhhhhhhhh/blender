from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *
import cv2
import ctypes
import numpy as np
from PIL import Image, ImageOps
import json


import sys
import os
# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Add the current script's directory to the Python path
sys.path.append(current_script_directory)
print(current_script_directory)


ip_addr = "169.254.8.21"
stereo_parameters_file = os.path.join(current_script_directory, "StereoWithExternal2DSystemParams.json")
# stereo_parameters_file = os.path.join(current_script_directory, "TrinoSystemParams.json")
Q_file = os.path.join(current_script_directory,"Q.json")
P1_file = os.path.join(current_script_directory,"P1.json") 
P2_file = os.path.join(current_script_directory,"P2.json")
cam_file = os.path.join(current_script_directory,"camera.json")

target_size = (640, 480)
# target_size = (1080, 960)

def print_func():
    print("Test")

def capture_by_stereo():
    sc = StereoCamera()
    sc.connect()
    sc.read_stereo_parameters(stereo_parameters_file)
    sc.expose_time(300)
    rectified_left, rectified_right = sc.capture_stereo_images(resize=True)
    sc.disconnect()
    return rectified_left, rectified_right


# def read_stereo_parameters(stereo_parameter_file):
#     import json
#     with open(stereo_parameter_file) as f:
#         stereo_paras = json.load(f)
#     R = stereo_paras['cam1_to_cam0_r'] 
#     t = stereo_paras['cam1_to_cam0_t'] 
#     cam0_dist_coeff = stereo_paras['set0']['camera']['dist_coefficients']
#     cam0_intrinsic = stereo_paras['set0']['camera']['intrinsic']
#     cam1_dist_coeff = stereo_paras['set1']['camera']['dist_coefficients']
#     cam1_intrinsic = stereo_paras['set1']['camera']['intrinsic']

#     return R, t, cam0_dist_coeff, cam0_intrinsic, cam1_dist_coeff, cam1_intrinsic



class StereoCamera(object):
    def __init__(self):
        self.camera = Camera()

    def disconnect(self):
        self.camera.disconnect()
        print("Disconnected from the camera successfully.")

    
    def connect(self):
        print("Discovering all available cameras...")
        # camera_infos = Camera.discover_cameras()

        # if len(camera_infos) == 0:
        #     print("No cameras found.")
        #     return

        # # Display the information of all available cameras.
        # for i in range(len(camera_infos)):
        #     print("Camera index :", i)
        #     print_camera_info(camera_infos[i])

        # input_index = 0

        # error_status = self.camera.connect(camera_infos[input_index])
        # if not error_status.is_ok():
        #     show_error(error_status)
        #     return
        
        # self.camera.get_camera_intrinsics(self.intrinsics)
        # print_camera_intrinsics(self.intrinsics)
        # print(self.intrinsics.texture.camera_matrix.fx)

        error_status = self.camera.connect(ip_addr)
        if not error_status.is_ok():
            show_error(error_status)
            return
        print("Connected to the camera successfully.")
    
    def expose_time(self, value=200):
        # Obtain the name of the currently selected user set.
        current_user_set = self.camera.current_user_set()
        error, user_set_name = current_user_set.get_name()
        show_error(error)
        print("\ncurrent_user_set: " + user_set_name)


        # Set the exposure mode and exposure time for capturing the 2D image, and then obtain the
        # parameter values to check if the setting was successful.
        exposure_mode_2d = Scanning2DExposureMode.Value_Timed
        error = current_user_set.set_enum_value(
            Scanning2DExposureMode.name, exposure_mode_2d)
        show_error(error)
        exposure_time_2d = value
        error = current_user_set.set_float_value(
            Scanning2DExposureTime.name, exposure_time_2d)
        show_error(error)
        error, exposure_mode_2d = current_user_set.get_enum_value_string(
            Scanning2DExposureMode.name)
        show_error(error)
        error, exposure_time_2d = current_user_set.get_float_value(
            Scanning2DExposureTime.name)
        show_error(error)
        print("\n2D scanning exposure mode enum: {}, exposure time: {}".
              format(exposure_mode_2d, exposure_time_2d))
    
    def update_image(self, K, image, target_size=target_size):
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
        
        def scale_image(image, target_size):
            """
            Scales an image to the target size.

            :param image: PIL.Image object
            :param target_size: tuple (width, height)
            :return: new PIL.Image object with the target size
            """
            return image.resize(target_size, Image.Resampling.LANCZOS)

        def update_K(intrinsic_matrix, original_size, target_size, padding):
            """
            Updates the intrinsic camera parameters based on scaling and padding.

            :param intrinsic_matrix: numpy array of intrinsic camera parameters
            :param original_size: tuple (original_width, original_height)
            :param target_size: tuple (target_width, target_height)
            :param padding: tuple (left, top, right, bottom)
            :return: new numpy array of updated intrinsic camera parameters
            """
            original_width, original_height = original_size
            target_width, target_height = target_size

            # Calculate scaling factors
            scale_x = target_width / (original_width + padding[0] + padding[2])
            scale_y = target_height / (original_height + padding[1] + padding[3])

            # in my case scale_x and scale_y should be the same
            # print(scale_x, scale_y)
            # print(padding)

            # Create a scaling matrix
            scaling_matrix = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ])

            # Update the intrinsic matrix
            updated_intrinsic_matrix = np.dot(scaling_matrix, intrinsic_matrix)

            # Adjust the principal point by the padding
            updated_intrinsic_matrix[0, 2] += padding[0] * scale_x
            updated_intrinsic_matrix[1, 2] += padding[1] * scale_y

            return updated_intrinsic_matrix
        
        new_image, padding = padding_image(image, target_ratio=target_size[0]/target_size[1])
        new_image = scale_image(new_image, target_size)
        new_K = update_K(K, image.size, target_size, padding)
        return new_image, new_K
        
    def read_stereo_parameters(self, stereo_parameter_file):
        with open(stereo_parameter_file) as f:
            stereo_paras = json.load(f)

        R = stereo_paras['cam0_to_cam1_r'] 
        t = stereo_paras['cam0_to_cam1_t'] 
        cam0_dist_coeff = stereo_paras['camera0']['dist_coefficients']
        cam0_intrinsic = stereo_paras['camera0']['intrinsic']
        cam1_dist_coeff = stereo_paras['camera1']['dist_coefficients']
        cam1_intrinsic = stereo_paras['camera1']['intrinsic']
        
        # # mech eye sdk 2.3+
        # R = stereo_paras['cam1_to_cam0_r'] 
        # t = stereo_paras['cam1_to_cam0_t'] 
        # cam0_dist_coeff = stereo_paras['set0']['camera']['dist_coefficients']
        # cam0_intrinsic = stereo_paras['set0']['camera']['intrinsic']
        # cam1_dist_coeff = stereo_paras['set1']['camera']['dist_coefficients']
        # cam1_intrinsic = stereo_paras['set1']['camera']['intrinsic']

        # # mech eye sdk 2.1
        # R = stereo_paras['slaveCamera']['to_master_r'] 
        # t = stereo_paras['slaveCamera']['to_master_t'] 
        # cam0_dist_coeff = stereo_paras['masterCamera']['dist_coefficients']
        # cam0_intrinsic = stereo_paras['masterCamera']['intrinsic']
        # cam1_dist_coeff = stereo_paras['slaveCamera']['camera_intri']['dist_coefficients']
        # cam1_intrinsic = stereo_paras['slaveCamera']['camera_intri']['intrinsic']
        

        self.R = np.array(R).reshape(3,3)
        self.t = np.array(t) * 0.001  # mm to m
        self.left_dist_coeff = np.array(cam0_dist_coeff)
        self.right_dist_coeff = np.array(cam1_dist_coeff)
        left_K = np.identity(3)
        left_K[0][0] = cam0_intrinsic[0]
        left_K[1][1] = cam0_intrinsic[1]
        left_K[0][2] = cam0_intrinsic[2]
        left_K[1][2] = cam0_intrinsic[3]
        right_K = np.identity(3)
        right_K[0][0] = cam1_intrinsic[0]
        right_K[1][1] = cam1_intrinsic[1]
        right_K[0][2] = cam1_intrinsic[2]
        right_K[1][2] = cam1_intrinsic[3]
        self.left_K = left_K
        self.right_K = right_K
    
    def construct_numpy_array_from_ctypes(self, raw_data_pointer, height, width, channels):
        """
        Construct a NumPy array from a raw data pointer using ctypes.

        Parameters:
        raw_data_pointer (ctypes.POINTER(ctypes.c_ubyte)): Pointer to the raw image data.
        height (int): Height of the image.
        width (int): Width of the image.
        channels (int): Number of color channels in the image.

        Returns:
        np.ndarray: NumPy array representing the image.
        """
        size = height * width * channels
        raw_data_array = np.ctypeslib.as_array(raw_data_pointer, shape=(size,))
        image_array = raw_data_array.reshape((height, width, channels))

        return image_array
   
    def capture_stereo_images(self, resize=False):
        stereo_left = Frame2D()
        stereo_right = Frame2D()
        error = self.camera.capture_stereo_2d(stereo_left, stereo_right)
        if not error.is_ok():
            show_error(error)
            return
        if stereo_left.color_type() == ColorTypeOf2DCamera_Monochrome:
            image_left = stereo_left.get_gray_scale_image()
            image_right = stereo_right.get_gray_scale_image()
        elif stereo_right.color_type() == ColorTypeOf2DCamera_Color:
            image_left = stereo_left.get_color_image()
            image_right = stereo_right.get_color_image()

        image_left = self.construct_numpy_array_from_ctypes(image_left.data(), image_left.height(), image_left.width(), 1)
        image_right = self.construct_numpy_array_from_ctypes(image_right.data(), image_right.height(), image_right.width(), 1)
        image_shape = image_left.shape
        image_left = image_left.reshape((image_shape[0], image_shape[1]))
        image_right = image_right.reshape((image_shape[0], image_shape[1]))

        if resize:
            image_left, self.left_K = self.update_image(self.left_K, Image.fromarray(image_left), target_size=target_size)
            image_right, self.right_K = self.update_image(self.right_K, Image.fromarray(image_right), target_size=target_size)
            image_left = np.array(image_left)
            image_right = np.array(image_right)

        # print(image_right)
        # print(image_right.shape)
        rectified_left, rectified_right, Q = self.rectify_stereo_images(image_left, image_right)
        # image_file_left = "stereo2D_left.png"
        # image_file_right = "stereo2D_right.png"
        # # # cv2.imshow(image_file_left, image_left.data())
        # # # cv2.imshow(image_file_right, image_right.data())
        # # cv2.waitKey(0)

        # cv2.imwrite(image_file_left, image_left)
        # cv2.imwrite(image_file_right, image_right)
        # print("Capture and save the stereo 2D images: {} and {}".format(
        #     image_file_left, image_file_right))
        
        # return image_left, image_right
        return rectified_left, rectified_right
    
    def rectify_stereo_images(self, left_rgb_image, right_rgb_image):
        w, h = left_rgb_image.shape[1], left_rgb_image.shape[0]
        # print(w, h)
        # print(self.left_K)
        # print(self.left_dist_coeff)
        # print(self.right_K)
        # print(self.right_dist_coeff)
        # print(self.R)
        # print(self.t)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(self.left_K, self.left_dist_coeff, self.right_K, self.right_dist_coeff, (w, h), self.R, self.t)
        map1x, map1y = cv2.initUndistortRectifyMap(self.left_K, self.left_dist_coeff, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.right_K, self.right_dist_coeff, R2, P2, (w, h), cv2.CV_32FC1)
        rectified_left_rgb = cv2.remap(left_rgb_image, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        rectified_right_rgb = cv2.remap(right_rgb_image, map2x, map2y, interpolation=cv2.INTER_LINEAR)
        image_file_left = os.path.join(current_script_directory, "stereo2D_left.png")
        image_file_right = os.path.join(current_script_directory, "stereo2D_right.png")
        # # cv2.imshow(image_file_left, image_left.data())
        # # cv2.imshow(image_file_right, image_right.data())
        # cv2.waitKey(0)

        cv2.imwrite(image_file_left, rectified_left_rgb)
        cv2.imwrite(image_file_right, rectified_right_rgb)

        print(Q)
        with open(Q_file, 'w') as file:
            json.dump(Q.tolist(), file)

        print(P1)
        with open(P1_file, 'w') as file:
            json.dump(P1.tolist(), file)

        print(P2)
        with open(P2_file, 'w') as file:
            json.dump(P2.tolist(), file)

        cam_json = {
            "width": target_size[0],
            "height": target_size[1],
            "depth_scale": 1,
            "cx": P1[0][2],
            "cy": P1[1][2],
            "fx": P1[0][0],
            "fy": P1[1][1],
            "baseline": 1/Q[3][2]
        }

        print(cam_json)
        with open(cam_file, 'w') as file:
            json.dump(cam_json, file)

        return rectified_left_rgb, rectified_right_rgb, Q



if __name__ == '__main__':
    a = StereoCamera()
    a.connect()
    a.read_stereo_parameters(stereo_parameters_file)
    a.expose_time(80)
    a.capture_stereo_images(resize=True)