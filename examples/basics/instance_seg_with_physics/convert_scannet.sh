#!/bin/bash

BLENDER_PROC_PATH=/home/fanxiaochen/projects/BlenderProc/
BOP_MODEL_PATH=/home/fanxiaochen/projects/BOP/BOP_dataset_sim

SIM_DATA=$1
DATA_START=$2
DATA_END=$3
OUTPUT_PATH=$4
BACKGROUND=$5

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
fi

for i in $(seq $DATA_START $DATA_END); do
    scene_id="$SIM_DATA/Scene_$i"
    #scene_id="$SIM_DATA/Scene_$(($i-1))"
    echo $scene_id
    python generate_gt.py ${BLENDER_PROC_PATH}/examples/basics/instance_seg_with_physics/${scene_id}/bop_data/lm/camera.json ${BLENDER_PROC_PATH}/examples/basics/instance_seg_with_physics/${scene_id}/bop_data/lm/train_pbr/000000/ $i ${OUTPUT_PATH} ${BACKGROUND}
done

