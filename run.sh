#python examples/basics/instance_seg_with_physics/generate_gt.py examples/basics/instance_seg_with_physics/output/bop_data/lm/camera.json examples/basics/instance_seg_with_physics/output/bop_data/lm/train_pbr/000000/depth/000012.png examples/basics/instance_seg_with_physics/output/0.hdf5

#python examples/basics/instance_seg_with_physics/generate_gt.py examples/basics/instance_seg_with_physics/output/bop_data/lm/camera.json examples/basics/instance_seg_with_physics/output/bop_data/lm/train_pbr/000000/depth/000012.png examples/basics/instance_seg_with_physics/output/0.hdf5 "scene000" examples/basics/instance_seg_with_physics/output

blenderproc run examples/basics/instance_seg_with_physics/main.py ../BOP_dataset/ "lm" resources/cc0_textures 9 20 examples/basics/instance_seg_with_physics/output
