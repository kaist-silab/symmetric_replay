#!/bin/bash 

oracle_array=('thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' )

for seed in 0
do
for oralce in "${oracle_array[@]}"
do
# echo $oralce
CUDA_VISIBLE_DEVICES=4 python run.py reinvent_selfies --task simple --oracle $oralce --wandb online --run_name rnd --seed $seed --config_default hparams_symrd.yaml
done
done
