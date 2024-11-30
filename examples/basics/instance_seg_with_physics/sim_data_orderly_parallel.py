import subprocess
import multiprocessing as mp
from tqdm import tqdm
import argparse
import random


NUMBER_OF_TASKS = 1
progress_bar = tqdm(total=NUMBER_OF_TASKS)


def work(sim_id, folder, scene_id, grid_x, grid_y, grid_z, grid_x_delta, grid_y_delta, grid_z_delta, model_mm, obj_id, blenderproc_path, bop_model_path):
    command = "bash sim_data_orderly.sh sim{}/{} {} {} {} {} {} {} {} {} {} {} {} {}".format(
        sim_id, folder, scene_id, 
        grid_x, grid_y, grid_z,
        grid_x_delta,grid_y_delta,grid_z_delta,
        model_mm, sim_id, obj_id,
        blenderproc_path, bop_model_path)
    print(command)
    subprocess.check_call(command, shell=True)


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
    parser.add_argument('model_mm', nargs='?', default="True", help="mm unit of model")
    parser.add_argument('grid_x', nargs='?', default="1", help="Grid x of objects for physics positioning")
    parser.add_argument('grid_y', nargs='?', default="1", help="Grid y of objects for physics positioning")
    parser.add_argument('grid_z', nargs='?', default="1", help="Grid z of objects for physics positioning")
    parser.add_argument('grid_x_delta', nargs='?', default="0.1", help="Grid x delta of objects for physics positioning")
    parser.add_argument('grid_y_delta', nargs='?', default="0.1", help="Grid y delta of objects for physics positioning")
    parser.add_argument('grid_z_delta', nargs='?', default="0.1", help="Grid z delta of objects for physics positioning")
    parser.add_argument('obj_range_begin', nargs='?', default="9", help="the end of obj id range")
    parser.add_argument('obj_range_end', nargs='?', default="9", help="the end of obj id range")
    args = parser.parse_args()
    pool = mp.Pool(NUMBER_OF_TASKS)

    for scene_id in range(int(args.scene_start), int(args.scene_start)+int(args.scene_num)):
        pool.apply_async(work, (args.sim_id, args.folder, scene_id, 
        args.grid_x, args.grid_y,args.grid_z,
        args.grid_x_delta,args.grid_y_delta,args.grid_z_delta, 
        args.model_mm, random.randint(int(args.obj_range_begin), int(args.obj_range_end)), args.blenderproc_path, args.bop_model_path), callback=update_progress_bar)

    pool.close()
    pool.join()
