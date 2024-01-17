#!/bin/bash 

oracle_array=('ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo')

for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=7 python run.py graph_ga --task simple --oracle $oralce --wandb online --run_name graph_ga --seed 0
done
