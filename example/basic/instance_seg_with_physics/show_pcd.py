import json
import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import open3d as o3d
import numpy as np

def convert_depth_to_points(intri, depth_img, depth_trunc=10000, label=None, use_color=False):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intri['width'], intri['height'], intri['fx'], intri['fy'], intri['cx'], intri['cy'])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img, intrinsic=intrinsic, extrinsic=np.identity(4),
        depth_scale=1000, depth_trunc=depth_trunc, stride=int(1))
    
    if label is not None:
        if use_color:
            flat_label = label.flatten()
            label_num = np.max(flat_label)
            colors = [color_map(index/label_num) for index in flat_label]
        else:
            flat_label = label.flatten()
            #label_num = np.max(flat_label)
            #colors = [color_map(index/label_num) for index in flat_label]
            #print(colors)
            colors = np.zeros((flat_label.shape[0], 3), dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    
   # print(np.max(depth_img))
   # print(np.min(depth_img))
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
    #o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("show.ply", pcd)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    #vis.get_render_option().background_color = [0,0,0]
    vis.run()
    vis.destroy_window()


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
                "label": label_name
            }
        )
    agg_path = os.path.join(output_dir, "{}.aggregation.json".format(gt_id))
    with open(agg_path, 'w') as f:
        json.dump(aggregation, f, indent=4)



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
    label = read_hdf5(label_img).flatten()
    pcd = convert_depth_to_points(intrinsic, depth_raw, depth_trunc=10000 , label=label)
    #pcd, label = remove_and_sample(pcd, label, 30000, [7, 8, 9, 10, 11], has_color=True)
    bg_indice = get_scene_label_ids(label)
    pcd, label = remove_and_sample(pcd, label, 30000, bg_indice , has_color=True)

    # write point cloud as ply
    o3d.io.write_point_cloud(os.path.join(output_dir, "{}_vh_clean_2.ply".format(gt_id)), pcd)
    write_segs(gt_id, label, output_dir)
    write_aggregation(gt_id, "duck", bg_indice[0]-1, output_dir)

def get_scene_label_ids(label):
    max_idx = np.max(label)
    # support current scene has 5 wall
    return range(max_idx-4, max_idx+1)


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

def dbscan_cluster():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5py', nargs='?', help="Path to the h5 file")
    args = parser.parse_args()

    with h5py.File(args.h5py, 'r') as hf:
        points = hf['points'][:]
        colors = np.zeros_like(points)
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.006, min_samples=10).fit(points).labels_
        print(clustering)
        labels = np.unique(clustering)
        label_num = labels.shape[0]
        for i in range(label_num):
            l = i - 1
            if l != -1:
                colors[clustering == l] = color_map(l/label_num)
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    #vis.get_render_option().background_color = [0,0,0]
    vis.run()
    vis.destroy_window()

def generate_gt():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_intrinsic', nargs='?', help="Path to the camera instrinsic json")
    parser.add_argument('train_pbr', nargs='?', help="Path to the rendering image")
    parser.add_argument('train_id', nargs='?', help="Train sample id")
    parser.add_argument('output_dir', nargs='?', help="Path to save gt data")
    args = parser.parse_args()

    scene = {
        "camera_intrinsic": args.camera_intrinsic,
        "gt_id": str(args.train_id),
        "depth_img": os.path.join(args.train_pbr, "depth", "000000.png"),
        "label_img": os.path.join(args.train_pbr, "0.hdf5")
    }

    generate_large_scannet_gt(scene, args.output_dir)


#generate_scannet_gt()

#show_points()

#generate_gt()

show_h5py()

#dbscan_cluster()


