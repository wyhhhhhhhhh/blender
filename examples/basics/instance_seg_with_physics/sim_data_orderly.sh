#!/bin/bash

DATA_PATH=$1
TOTAL_NUM=$2
GRID_X=$3
GRID_Y=$4
GRID_Z=$5
GRID_X_DELTA=$6
GRID_Y_DELTA=$7
GRID_Z_DELTA=$8
MODEL_MM=$9
SIM_ID=${10}
OBJ_ID=${11}
BLENDERPROC_PATH=${12}
BOP_MODEL_PATH=${13}

if [ ! -d "$DATA_PATH" ]; then
    mkdir -p $DATA_PATH
fi

#for i in $(seq 1 $TOTAL_NUM); do
#    scene_id="$DATA_PATH/Scene_$i"
#    mkdir ${scene_id}
#    echo $i;
#    blenderproc run /home/fanxiaochen/projects/BlenderProc/examples/basics/instance_seg_with_physics/main.py /home/fanxiaochen/projects/BOP/BOP_dataset_sim${SIM_ID}/ "lm" ${MODEL_MM} /home/fanxiaochen/projects/BlenderProc/resources/cc0_textures 9 ${GRID_X} ${GRID_Y} ${GRID_Z} ${GRID_X_DELTA} ${GRID_Y_DELTA} ${GRID_Z_DELTA} ${scene_id}
#done


generate(){
    scene_id="$DATA_PATH/Scene_$1"
    mkdir ${scene_id}
    echo $1;
    blenderproc run ${BLENDERPROC_PATH}examples/basics/instance_seg_with_physics/main.py ${BOP_MODEL_PATH}${SIM_ID}/ "lm" ${MODEL_MM} ${BLENDERPROC_PATH}resources/cc0_textures ${OBJ_ID} ${GRID_X} ${GRID_Y} ${GRID_Z} ${GRID_X_DELTA} ${GRID_Y_DELTA} ${GRID_Z_DELTA} ${scene_id}
}

generate $TOTAL_NUM