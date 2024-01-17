#!/bin/bash 

oracle_array=('sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts')


for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=7 python run.py graph_ga --task simple --oracle $oralce --wandb online --run_name graph_ga --seed 0
done