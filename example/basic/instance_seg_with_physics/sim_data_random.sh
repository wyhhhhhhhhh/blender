#!/bin/bash

DATA_PATH=$1
TOTAL_NUM=$2
GEN_NUM=$3
SIM_ID=$4
MODEL_MM=$5
OBJ_ID=$6
BACKGROUND=$7
BOX_WIDTH=$8
BLENDERPROC_PATH=$9
BOP_MODEL_PATH=${10}

if [ ! -d "$DATA_PATH" ]; then
    mkdir -p $DATA_PATH
fi


#for i in $(seq 1 $TOTAL_NUM); do
#    scene_id="$DATA_PATH/Scene_$i"
#    mkdir ${scene_id}
#    echo $i;
#    blenderproc run /home/fanxiaochen/projects/BlenderProc/examples/basics/instance_seg_with_physics/seg_generator.py /home/fanxiaochen/projects/BOP/BOP_dataset_sim${SIM_ID}/ "lm" /home/fanxiaochen/projects/BlenderProc/resources/cc0_textures 9 ${GEN_NUM} ${scene_id}
#done

generate(){
    scene_id="$DATA_PATH/Scene_$1"
    mkdir ${scene_id}
    echo $1;
    blenderproc run ${BLENDERPROC_PATH}examples/basics/instance_seg_with_physics/seg_generator.py ${BOP_MODEL_PATH}${SIM_ID}/ "lm" ${MODEL_MM} ${BLENDERPROC_PATH}resources/cc0_textures ${OBJ_ID} ${GEN_NUM} ${scene_id} ${BACKGROUND} ${BOX_WIDTH}
}

#for i in $(seq 1 $TOTAL_NUM); do
#	generate $i 
#done
generate $TOTAL_NUM
