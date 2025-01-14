import subprocess
import os
import multiprocessing as mp
from tqdm import tqdm
import argparse
import random
import logging

def call_random(**kargs):
   # print(kargs)
   # print(kargs["folder"])
   # command = "python sim_data_random_parallel.py train 1 30 50 5 False 3 3"
    command = "python sim_data_random_parallel.py {} {} {} {} {} {} {} {} {} {} {} {}".format(
        kargs["blenderproc_path"], kargs["bop_model_path"], 
        kargs["folder"], kargs["scene_start"], kargs["scene_num"], kargs["sim_id"], kargs["obj_num"], 
        kargs["model_mm"], kargs["obj_range_begin"], kargs["obj_range_end"], kargs["background"], ' '.join(kargs["box_width"]))
    print(command)
    subprocess.check_call(command, shell=True)
    pass

def call_ordered(**kargs):
   # command = "python sim_data_orderly_parallel.py train 1 60 47 False 4 4 3 0.01 0.01 0.01 3 3"
    command = "python sim_data_orderly_parallel.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
        kargs["blenderproc_path"], kargs["bop_model_path"], 
        kargs["folder"], kargs["scene_start"], kargs["scene_num"], kargs["sim_id"], 
        kargs["model_mm"], kargs["grid_x"], kargs["grid_y"], kargs["grid_z"], 
        kargs["grid_x_delta"], kargs["grid_y_delta"], kargs["grid_z_delta"],
        kargs["obj_range_begin"], kargs["obj_range_end"])
    print(command)
    subprocess.check_call(command, shell=True)

def call_background(**kargs):
   # command = "python sim_data_random_parallel.py train 1 30 50 5 False 3 3"
    command = "python sim_data_random_parallel.py {} {} {} {} {} {} {} {} {} {} {} {}".format(
        kargs["blenderproc_path"], kargs["bop_model_path"], 
        kargs["folder"], kargs["scene_start"], kargs["scene_num"], kargs["sim_id"], kargs["obj_num"], 
        kargs["model_mm"], kargs["obj_range_begin"], kargs["obj_range_end"], kargs["background"], ' '.join(kargs["box_width"]))
    print(command)
    subprocess.check_call(command, shell=True)

call_functions = {  'random': call_random,
                    'ordered': call_ordered, 
                    'background': call_background}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('blenderproc_path', nargs='?', help="Path to blenderproc")
    parser.add_argument('bop_model_path', nargs='?', help="Path to bop_model")
    parser.add_argument('folder', nargs='?', help="Train or val")
    parser.add_argument('mode', nargs='?', default="random", help="random, ordered, background")
    parser.add_argument('sim_id', nargs='?', default=1, help="Sim id", type=int)
    parser.add_argument('scene_start', nargs='?', default=1, help="Start scene id", type=int)
    parser.add_argument('scene_num', nargs='?', default=1, help="Number of scenes", type=int)

    parser.add_argument('obj_range_begin', nargs='?', default=1, help="the start of obj id range", type=int)
    parser.add_argument('obj_range_end', nargs='?', default=1, help="the end of obj id range", type=int)
    parser.add_argument('model_mm', nargs='?', default="False", help="unit of model")

    # for random
    parser.add_argument('--obj_num', nargs='?', default=1, help="Number of objects for physics positioning", type=int)
    parser.add_argument('--box_width', nargs='+', default="0.5", help="Box width of background")
    # for ordered
    parser.add_argument('--grid_x', nargs='?', default=1, help="Grid x of objects for physics positioning", type=int)
    parser.add_argument('--grid_y', nargs='?', default=1, help="Grid y of objects for physics positioning", type=int)
    parser.add_argument('--grid_z', nargs='?', default=1, help="Grid z of objects for physics positioning", type=int)
    parser.add_argument('--grid_x_delta', nargs='?', default=0.1, help="Grid x delta of objects for physics positioning", type=float)
    parser.add_argument('--grid_y_delta', nargs='?', default=0.1, help="Grid y delta of objects for physics positioning", type=float)
    parser.add_argument('--grid_z_delta', nargs='?', default=0.1, help="Grid z delta of objects for physics positioning", type=float)
    # for background
    parser.add_argument('--background', nargs='?', default="False", help="Generating background data")

    args = parser.parse_args()

    print(vars(args))
    call_functions[args.mode](**vars(args))
