#!/bin/bash

BLENDER_PROC_PATH=/mnt/d/blenderproc/
BOP_MODEL_PATH=/mnt/d/blenderproc/BOP_dataset_sim

SIM_ID=3
OBJ_START_ID=0
OBJ_END_ID=0
MODEL_MM=False


# ### random
# python sim_data_generator.py ${BLENDER_PROC_PATH} ${BOP_MODEL_PATH} train random ${SIM_ID} 1 50 ${OBJ_START_ID} ${OBJ_END_ID} ${MODEL_MM} --obj_num 40 --box_width 0.4
# python sim_data_generator.py ${BLENDER_PROC_PATH} ${BOP_MODEL_PATH} val random ${SIM_ID} 1 5 ${OBJ_START_ID} ${OBJ_END_ID} ${MODEL_MM} --obj_num 40 --box_width 0.4

# background
python sim_data_generator.py ${BLENDER_PROC_PATH} ${BOP_MODEL_PATH} train background ${SIM_ID} 1 500 ${OBJ_START_ID} ${OBJ_END_ID} ${MODEL_MM} --background True --obj_num 20  --box_width 0.3
python sim_data_generator.py ${BLENDER_PROC_PATH} ${BOP_MODEL_PATH} val background ${SIM_ID} 1 20 ${OBJ_START_ID} ${OBJ_END_ID} ${MODEL_MM} --background True --obj_num 20 --box_width 0.3

# # convert to scannet
# bash convert_scannet.sh sim${SIM_ID}/train 1 50 sim${SIM_ID}/train_scannet 
# bash convert_scannet.sh sim${SIM_ID}/val 1 5 sim${SIM_ID}/val_scannet

# bash convert_scannet.sh sim${SIM_ID}/train 51 70 sim${SIM_ID}/train_scannet --background True
# bash convert_scannet.sh sim${SIM_ID}/val 6 7 sim${SIM_ID}/val_scannet --background True


cp $0 sim${SIM_ID}
