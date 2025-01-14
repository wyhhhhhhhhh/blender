import json
import argparse
import os
from tkinter import Scrollbar
import h5py
import open3d as o3d
import numpy as np

#import debugpy
#debugpy.listen(5678)
#debugpy.wait_for_client()

def convert_depth_to_points(intri, depth_img, depth_trunc=10000, label=None, use_color=False):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intri['width'], intri['height'], intri['fx'], intri['fy'], intri['cx'], intri['cy'])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img, intrinsic=intrinsic, extrinsic=np.identity(4),
        depth_scale=1000, depth_trunc=depth_trunc, stride=int(1))

  # # remove null depth, value=65535 
  #  xyz = np.asarray(pcd.points)
  #  valid_idx = np.where(xyz[:,2] < 65)
  #  print(len(valid_idx[0]))
  #  pcd_v = o3d.geometry.PointCloud()
  #  pcd_v.points = o3d.utility.Vector3dVector(xyz[valid_idx])

    
    if label is not None:
        if use_color:
            flat_label = label.flatten()
            #print("label:", np.min(flat_label), np.max(flat_label))
            #label_num = np.max(flat_label)
            flat_s = np.unique(flat_label)
            label_num = np.count_nonzero(flat_s < 1000)
            print(label_num)
            colors = [color_map(index/label_num) if index < 1000 else [0,0,0] for index in flat_label]
  #          colors = colors[valid_idx]
        else:
            flat_label = label.flatten()
            #label_num = np.max(flat_label)
            #colors = [color_map(index/label_num) for index in flat_label]
            #print(colors)
            colors = np.zeros((flat_label.shape[0], 3), dtype=np.float32)
  #          colors = colors[valid_idx]
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    
  #  print(np.max(depth_img))
  #  print(np.min(depth_img))
    # print(pcd.points)
    return pcd

def read_camera_intrinsic(instrinsic_json):
    with open(instrinsic_json, 'r') as f:
        intrinsic = json.load(f)
    #print(intrinsic)
    return intrinsic

def read_raw_depth(depth_img_path):
    depth_raw = o3d.io.read_image(depth_img_path)
    np_depth = np.array(depth_raw)
   # print(np.max(np_depth))
   # print(np.min(np_depth))
   # print(np_depth)
    depth_raw = o3d.geometry.Image(np_depth)
    return depth_raw

def read_label_img(label_img_path):
    filename = label_img_path
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        #print(np.array(f["instance_segmaps"][:]))
        label_img = np.array(f["instance_segmaps"][:])
        return label_img

def read_hdf5(hdf5_path):
    filename = hdf5_path
    with h5py.File(filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        label_img = np.array(f["instance_segmaps"][:])
       # print(np.max(label_img))
       # print(np.min(label_img))
        return label_img

def read_hdf5_with_scores(hdf5_path):
    filename = hdf5_path
    with h5py.File(filename, "r") as f:
        # List all groups
       # print("Keys: %s" % f.keys())
        label_img = np.array(f["instance_segmaps"][:])
        score_map = {}
        dict_group_load = f['instance_scores']
        dict_group_keys = dict_group_load.keys()
        for k in dict_group_keys:
            score_map[k]= dict_group_load[k][()]
        print("score_map:", score_map)
        #score_map = np.array(f["instance_scores"][:])
       # print(np.max(label_img))
       # print(np.min(label_img))
        return label_img, score_map

def color_map(value, cmap_name='rainbow', vmin=0, vmax=1):
    import matplotlib.cm as cm
    import matplotlib as matplotlib

    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def show_points():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_intrinsic', nargs='?', help="Path to the camera instrinsic json")
    parser.add_argument('depth_img', nargs='?', help="Path to the depth image")
    parser.add_argument('label_img', nargs='?', help="Path to the instance label")
    args = parser.parse_args()

    depth_raw = read_raw_depth(args.depth_img)
    intrinsic = read_camera_intrinsic(args.camera_intrinsic)
    label_img = read_hdf5(args.label_img)
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000, label=label_img, use_color=True)
    o3d.visualization.draw_geometries([pcd])

def show_h5py():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5py', nargs='?', help="Path to the h5 file")
    args = parser.parse_args()

    with h5py.File(args.h5py, 'r') as hf:
        points = hf['points'][:]
        colors = hf['colors'][:]
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# write segs json
def write_segs(gt_id, label_img, output_dir):
    segs = dict()
    segs["params"] = {
        # useless params for us, set by default
        "kThresh": "0.0001",
        "segMinVerts": "20",
        "minPoints": "750",
        "maxPoints": "30000",
        "thinThresh": "0.05",
        "flatThresh": "0.001",
        "minLength": "0.02",
        "maxLength": "1"
    }

    segs["sceneId"] = gt_id

    flat_label = label_img.flatten().tolist()
    #label_num = np.max(flat_label)
   # flat_label_min = np.min(flat_label)
   # print("flat_label_min:", flat_label_min)
    #colors = [color_map(index/label_num) for index in flat_label]
    segs["segIndices"] = flat_label
    segs_path = os.path.join(output_dir, "{}_vh_clean_2.0.010000.segs.json".format(gt_id))
    with open(segs_path, 'w') as f:
        json.dump(segs, f, indent=4)

# write aggregation json
def write_aggregation(gt_id, label_name, instance_num, output_dir):
    aggregation = dict()
    aggregation["sceneId"] = "scannet."+gt_id
    aggregation["appId"] = "Aggregator.v2"
    aggregation["segmentsFile"] = "scannet.{}_vh_clean_2.0.010000.segs.json".format(gt_id)
    aggregation["segGroups"] = list()
    for obj_idx in range(instance_num):
        aggregation["segGroups"].append(
            {
                "id": obj_idx,
                "objectId": obj_idx,
                "segments": [obj_idx+1],
                "label": label_name,
                "completion": 1.0
            }
        )
    agg_path = os.path.join(output_dir, "{}.aggregation.json".format(gt_id))
    with open(agg_path, 'w') as f:
        json.dump(aggregation, f, indent=4)

# write aggregation json
def write_aggregation_with_bg(gt_id, label_map, score_map, output_dir):
    aggregation = dict()
    aggregation["sceneId"] = "scannet."+gt_id
    aggregation["appId"] = "Aggregator.v2"
    aggregation["segmentsFile"] = "scannet.{}_vh_clean_2.0.010000.segs.json".format(gt_id)
    aggregation["segGroups"] = list()

    sem_label = dict()
#    sem_names = {
#        'complete': 0,
#        'non-complete': 1,
#        'background': 2
#    }

    sem_names = {
        'object': 0,
        'background': 1
    }

    segments = np.unique(label_map)
    for obj_idx, segment in enumerate(segments):
        label = 'none'
        if segment > 0 and segment < 1000:
            #label = 'complete'
            label = 'object'
       # elif segment > 1000 and segment < 2000:
       #     label = 'non-complete'
        else:
            label = 'background'

        aggregation["segGroups"].append(
            {
                "id": obj_idx,
                "objectId": obj_idx,
                "segments": [segment.item()],
                "label": label,
                "completion": score_map[str(segment.item())] # extra score for network
            }
        )

        sem_label[segment] = sem_names[label]

    agg_path = os.path.join(output_dir, "{}.aggregation.json".format(gt_id))
    with open(agg_path, 'w') as f:
        json.dump(aggregation, f, indent=4)
    
    return sem_label

def write_sem_label_ply(pcd, label_img, sem_map, gt_id, output_dir):
    flat_label = label_img.flatten().tolist()
    remap_label = [sem_map[l] for l in flat_label]
    points = np.asarray(pcd.points)
    labels = np.asarray(remap_label)
    print("labels:",labels)
   # print(labels.dtype, labels.shape)
   # print(points.dtype, points.shape)
    from plyfile import PlyData, PlyElement
    #import numpy.lib.recfunctions
    #vertices = numpy.lib.recfunctions.merge_arrays([points, labels])
    vertices = np.array([(points[i][0], points[i][1], points[i][2], labels[i]) for i in range(points.shape[0])], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')])
    ply_out = PlyData([PlyElement.describe(vertices, 'vertex')])
    filename = os.path.join(output_dir, "{}_vh_clean_2.labels.ply".format(gt_id))
    ply_out.write(filename)


def generate_scannet_gt():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_intrinsic', nargs='?', help="Path to the camera instrinsic json")
    parser.add_argument('depth_img', nargs='?', help="Path to the depth image")
    parser.add_argument('label_img', nargs='?', help="Path to the instance label")
    parser.add_argument('gt_id', nargs='?', help="GT id for saving")
    parser.add_argument('output_dir', nargs='?', help="Path to save gt data")
    args = parser.parse_args()

    depth_raw = read_raw_depth(args.depth_img)
    intrinsic = read_camera_intrinsic(args.camera_intrinsic)
    label_img = read_hdf5(args.label_img)
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000)

    # write point cloud as ply
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "{}_vh_clean_2.ply".format(args.gt_id)), pcd)

    # write segs json
    def write_segs(gt_id, label_img):
        segs = dict()
        segs["params"] = {
            # useless params for us, set by default
            "kThresh": "0.0001",
            "segMinVerts": "20",
            "minPoints": "750",
            "maxPoints": "30000",
            "thinThresh": "0.05",
            "flatThresh": "0.001",
            "minLength": "0.02",
            "maxLength": "1"
        }

        segs["sceneId"] = gt_id

        flat_label = label_img.flatten().tolist()
        #label_num = np.max(flat_label)
        #colors = [color_map(index/label_num) for index in flat_label]
        segs["segIndices"] = flat_label
        segs_path = os.path.join(args.output_dir, "{}_vh_clean_2.0.010000.segs.json".format(gt_id))
        with open(segs_path, 'w') as f:
            json.dump(segs, f, indent=4)

    # write aggregation json
    def write_aggregation(gt_id, label_name, instance_num):
        aggregation = dict()
        aggregation["sceneId"] = "scannet."+gt_id
        aggregation["appId"] = "Aggregator.v2"
        aggregation["segmentsFile"] = "scannet.{}_vh_clean_2.0.010000.segs.json".format(gt_id)
        aggregation["segGroups"] = list()
        for obj_idx in range(instance_num):
            aggregation["segGroups"].append(
                {
                    "id": obj_idx,
                    "objectId": obj_idx,
                    "segments": [obj_idx+1],
                    "label": label_name
                }
            )
        agg_path = os.path.join(args.output_dir, "{}.aggregation.json".format(gt_id))
        with open(agg_path, 'w') as f:
            json.dump(aggregation, f, indent=4)


    # write labels json


    write_segs(args.gt_id, label_img)
    write_aggregation(args.gt_id, "duck", 6)



def generate_large_scannet_gt(scene, output_dir):
    gt_id = scene["gt_id"]
    camera_intrinsic = scene["camera_intrinsic"]
    depth_img = scene["depth_img"]
    label_img = scene["label_img"]

    depth_raw = read_raw_depth(depth_img)
    intrinsic = read_camera_intrinsic(camera_intrinsic)
    if not os.path.exists(label_img):
        return
    label = read_hdf5(label_img).flatten()
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000 , label=label)
    #pcd, label = remove_and_sample(pcd, label, 30000, [7, 8, 9, 10, 11], has_color=True)
    bg_indice = get_scene_label_ids(label)
    pcd, label = remove_and_sample(pcd, label, 30000, bg_indice , has_color=True)

    # write point cloud as ply
    o3d.io.write_point_cloud(os.path.join(output_dir, "{}_vh_clean_2.ply".format(gt_id)), pcd)
    write_segs(gt_id, label, output_dir)
    #write_aggregation(gt_id, "duck", bg_indice[0]-1, output_dir)
    write_aggregation(gt_id, "object", bg_indice[0]-1, output_dir)

def get_scene_label_ids(label):
    max_idx = np.max(label)
    # support current scene has 5 wall
    return range(max_idx-4, max_idx+1)

    # only one bg
    return range(max_idx, max_idx+1)


def remove_and_sample(pcd, label, sample_num, bg_index, has_color=False):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # remove background points
    bg_indices = list()
    for index in bg_index:
        indices = np.where(label == index)
        bg_indices += indices[0].tolist()
    # nbg_indices = list(np.where(label != bg_index))

    all_indices = np.arange(0, points.shape[0])
    nbg_indices = np.delete(all_indices, bg_indices)
    points = points[nbg_indices]
    label = label[nbg_indices]
    if has_color:
        colors = colors[nbg_indices]

    # randomly choose sampled points
    random_indices = np.random.choice(points.shape[0], sample_num)
    points = np.array(points)[random_indices]
    label = np.array(label)[random_indices]
    pcd.points = o3d.utility.Vector3dVector(points)
    if has_color:
        colors = np.array(colors)[random_indices]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    #o3d.visualization.draw_geometries([pcd])

    return pcd, label

def generate_large_scannet_gt_with_bg(scene, output_dir, background):
    gt_id = scene["gt_id"]
    camera_intrinsic = scene["camera_intrinsic"]
    depth_img = scene["depth_img"]
    label_img = scene["label_img"]

    depth_raw = read_raw_depth(depth_img)
    intrinsic = read_camera_intrinsic(camera_intrinsic)
    if not os.path.exists(label_img):
        return
    label, score_map = read_hdf5_with_scores(label_img)
    label = label.flatten()
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000 , label=label)
    pcd, label = sample(pcd, label, 150000, has_color=True, remove_bg=(background=="False"))

    # write point cloud as ply
    o3d.io.write_point_cloud(os.path.join(output_dir, "{}_vh_clean_2.ply".format(gt_id)), pcd)
    write_segs(gt_id, label, output_dir)
    sem_mapper = write_aggregation_with_bg(gt_id, label, score_map, output_dir)
    write_sem_label_ply(pcd, label, sem_mapper, gt_id, output_dir)


def sample(pcd, label, sample_num, has_color=False, remove_bg=False):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if remove_bg:
        # remove background points
        # label > 1000
        # max label used as random bg for training
        max_label = np.max(label)
        indices = np.where(label > 1000)
        #indices = np.where((label > 1001) & (label < max_label)) # remain bottom and random bg
        bg_indices = indices[0].tolist()

        all_indices = np.arange(0, points.shape[0])
        nbg_indices = np.delete(all_indices, bg_indices)
        points = points[nbg_indices]
        label = label[nbg_indices]
        if has_color:
            colors = colors[nbg_indices]

    # randomly choose sampled points
    if sample_num > points.shape[0]:
        random_indices = np.arange(0, points.shape[0])
    else:
        random_indices = np.random.choice(points.shape[0], sample_num, replace=False)
    points = np.array(points)[random_indices]
    label = np.array(label)[random_indices]
    pcd.points = o3d.utility.Vector3dVector(points)
    if has_color:
        colors = np.array(colors)[random_indices]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    #o3d.visualization.draw_geometries([pcd])

   # remove null depth, value=65535 
   # Due to little samples per pixel, many black pixels exist with null depth
   # Actually, the depth map should be individual with color map. Intersection between
   # ray and object should have exist, while the shading result may be null. 
    valid_idx = np.where(points[:,2] < 65)
    points = np.array(points)[valid_idx]
    label = np.array(label)[valid_idx]
    #print(len(valid_idx[0]))
    pcd.points = o3d.utility.Vector3dVector(points)
    if has_color:
        colors = np.array(colors)[valid_idx]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, label


def generate_gt():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_intrinsic', nargs='?', help="Path to the camera instrinsic json")
    parser.add_argument('train_pbr', nargs='?', help="Path to the rendering image")
    parser.add_argument('train_id', nargs='?', help="Train sample id")
    parser.add_argument('output_dir', nargs='?', help="Path to save gt data")
    parser.add_argument('--background', nargs='?', default="False", help="Generating background data")
    args = parser.parse_args()

    scene = {
        "camera_intrinsic": args.camera_intrinsic,
        "gt_id": str(args.train_id),
        "depth_img": os.path.join(args.train_pbr, "depth", "000000.png"),
        #"label_img": os.path.join(args.train_pbr, "0.hdf5")
        "label_img": os.path.join(args.train_pbr, "0_f.hdf5")
    }

    #generate_large_scannet_gt(scene, args.output_dir)
    generate_large_scannet_gt_with_bg(scene, args.output_dir, args.background)


#generate_scannet_gt()

#show_points()

generate_gt()

#show_h5py()



