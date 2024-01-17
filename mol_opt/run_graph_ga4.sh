#!/bin/bash 

oracle_array=('isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1')

for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=7 python run.py graph_ga --task simple --oracle $oralce --wandb online --run_name graph_ga --seed 0
done
