#!/bin/bash 

oracle_array=('thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' )

for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=7 python run.py graph_ga --task simple --oracle $oralce --wandb online --run_name graph_ga --seed 0
done
