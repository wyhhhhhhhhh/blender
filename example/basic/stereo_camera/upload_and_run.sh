#!/bin/bash

# Define variables
HOST="192.168.3.251"
USER="mechmind"
PASSWORD="mechmind"   # Note: It's not recommended to use passwords directly in scripts

REMOTE_ROOT="/data1/wyh/psmnet"
LOCAL_ROOT="/mnt/d/blenderproc/examples/basics/stereo_camera"
SIM_ID="sim3"

# run generation
sh sim_data_generator.sh
python rectify.py
TRAIN_ZIP=psm-${SIM_ID}-train
VAL_ZIP=psm-${SIM_ID}-val
zip -r ${TRAIN_ZIP}.zip ${TRAIN_ZIP}
zip -r ${VAL_ZIP}.zip ${VAL_ZIP}

# Upload the file using rsync
rsync -avz "${TRAIN_ZIP}.zip" "$USER@$HOST:$REMOTE_ROOT/dataset/$SIM_ID/train" --rsync-path="mkdir -p $REMOTE_ROOT/dataset/$SIM_ID/train && rsync"
rsync -avz "${VAL_ZIP}.zip" "$USER@$HOST:$REMOTE_ROOT/dataset/$SIM_ID/val" --rsync-path="mkdir -p $REMOTE_ROOT/dataset/$SIM_ID/val && rsync"

if [ $? -eq 0 ]; then
    echo "File uploaded successfully."
else
    echo "File upload failed."
    exit 1
fi

# # Execute the Python script on the remote server
# ssh -tt "$USER@$HOST" << EOF
# tmux kill-session -t mysession
# tmux new-session -d -s mysession
# tmux send-keys -t mysession "conda activate yolov8" C-m
# tmux send-keys -t mysession "cd ${REMOTE_ROOT}" C-m
# tmux send-keys -t mysession "sh unpack_data.sh ${SIM_ID}" C-m
# tmux send-keys -t mysession "python main.py --datapath dataset/sim3/ --epoch 300 --dataname sim3  --disp_min 288 --disp_max 400 --model stackhourglass --datatype 2" C-m

# EOF
ssh -tt "$USER@$HOST" << EOF
tmux kill-session -t mysession
tmux new-session -d -s mysession
conda activate yolov8
cd /data1/wyh/psmnet
sed -i 's/\r//' unpack_data.sh
sh unpack_data.sh sim3
python main.py --datapath dataset/sim3/ --epoch 300 --dataname sim3 --disp_min 416 --disp_max 528 --model stackhourglass --datatype 2
EOF

# conda activate psmnet
# cd ${REMOTE_ROOT}
# sh unpack_data.sh ${SIM_ID}
# python main.py --datapath dataset/${SIM_ID}/ --epochs 400 --dataname ${SIM_ID} --maxdisp 320 --model stackhourglass
# echo "Script completed successfully."

if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Failed to execute Python script."
    exit 1
fi
