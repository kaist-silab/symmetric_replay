for s in 123 1234 1 12 
do
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --batch_size 100 --epoch_size 10000 --n_epochs 200 --graph_size 50 --val_dataset '../data/tsp/tsp50_val_seed1234.pkl' --baseline critic --distil_every 1 --il_coefficient 0.001 --transform_opt adversarial --wandb online --run_name bl_val --distil_loop 1 --seed ${s}
done