import blenderproc as bproc
import argparse
import os
import random
import h5py
import numpy as np

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


def compare_mask_ratio_approximation(object_num, mixed_mask, label_idx):
    ## zero in mixed_mask means null value in depth map
    count_m = np.count_nonzero(mixed_mask == label_idx)
    full_mask = np.copy(mixed_mask)
    full_mask = full_mask[(np.where((full_mask <= object_num) & (full_mask > 0)))] # don't consider background, remove null values
    max_count = np.max(np.bincount(full_mask))
    return count_m/max_count

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