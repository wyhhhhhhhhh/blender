from genericpath import exists
import subprocess
import os
import multiprocessing as mp
from tqdm import tqdm
import argparse
import random

#def bash_func(data_path, scene_num, obj_num, sim_id, model_mm):
#    scene_id = os.path.join(data_path, "Scene_"+str(scene_num))
#    os.mkdir(scene_id)
#    command = "blenderproc run /home/fanxiaochen/projects/BlenderProc/examples/basics/instance_seg_with_physics/seg_generator.py /home/fanxiaochen/projects/BOP/BOP_dataset_sim{}/ \"lm\" {} /home/fanxiaochen/projects/BlenderProc/resources/cc0_textures 9 {} {}".format(
#        sim_id, model_mm, obj_num, scene_id)
#    subprocess.check_call(command, shell=True)
#    return "scene {} finished".format(scene_num)


NUMBER_OF_TASKS = 1
progress_bar = tqdm(total=NUMBER_OF_TASKS)


def work(sim_id, folder, scene_num, obj_num, model_mm, obj_id, background, box_width, blenderproc_path, bop_model_path):
    if background == "True":
        obj_num = random.randint(1, int(obj_num))
    command = "bash sim_data_random.sh sim{}/{} {} {} {} {} {} {} {} {} {}".format(sim_id, folder, scene_num, obj_num, sim_id, model_mm, obj_id, background, box_width, blenderproc_path, bop_model_path)
    print(command)
    subprocess.check_call(command, shell=True)
    return "scene {} finished".format(scene_num)

#    data_path = "sim{}/{}".format(sim_id, folder)
#    if not os.path.exists(data_path):
#        os.makedirs(data_path, exist_ok=True)
#    return bash_func(data_path, scene_num, obj_num, sim_id, model_mm)



def update_progress_bar(_):
    progress_bar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('blenderproc_path', nargs='?', help="Path to blenderproc")
    parser.add_argument('bop_model_path', nargs='?', help="Path to bop_model")
    parser.add_argument('folder', nargs='?', help="Train or val")
    parser.add_argument('scene_start', nargs='?', help="Start scene id")
    parser.add_argument('scene_num', nargs='?', help="Number of scenes ")
    parser.add_argument('sim_id', nargs='?', default="1", help="Sim id")
    parser.add_argument('obj_num', nargs='?', default="1", help="Number of objects for physics positioning")
    parser.add_argument('model_mm', nargs='?', help="unit of model")
    parser.add_argument('obj_range_begin', nargs='?', default="9", help="the end of obj id range")
    parser.add_argument('obj_range_end', nargs='?', default="9", help="the end of obj id range")
    parser.add_argument('background', nargs='?', default="False", help="Generating background data")
    parser.add_argument('box_width', nargs='+', default="0.5", help="Box width of background")
    args = parser.parse_args()
    pool = mp.Pool(NUMBER_OF_TASKS)

    results = []
    for scene_id in range(int(args.scene_start), int(args.scene_start)+int(args.scene_num)):
        obj_id = random.randint(int(args.obj_range_begin), int(args.obj_range_end))
        box_width = float(args.box_width[obj_id])
        res = pool.apply_async(work, (args.sim_id, args.folder, scene_id, args.obj_num, args.model_mm, obj_id,
        args.background, box_width, args.blenderproc_path, args.bop_model_path), callback=update_progress_bar)
        results.append(res)

    pool.close()
    pool.join()

    for i in results:
        print(i.get())
