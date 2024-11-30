import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', nargs='?', help="Path to the point clouds")
    parser.add_argument('input_start', nargs='?', help="input start id", type=int)
    parser.add_argument('input_end', nargs='?', help="input end id", type=int)
    parser.add_argument('offset', nargs='?', help="offset id", type=int)
    args = parser.parse_args()

    print(args.input_folder)
    if not os.path.exists(args.input_folder):
        exit(0)

    if args.offset > 0 and (args.input_start + args.offset) <= args.input_end:
        exit(0)
    if args.offset < 0 and (args.input_end + args.offset) >= args.input_start:
        exit(0)

    for idx in range(args.input_start, args.input_end+1):
        scene_file = "Scene_{}".format(idx)
        new_scene_file = "Scene_{}".format(idx+args.offset)
        if os.path.exists(os.path.join(args.input_folder, scene_file)):
            os.rename(os.path.join(args.input_folder, scene_file), os.path.join(args.input_folder, new_scene_file))
