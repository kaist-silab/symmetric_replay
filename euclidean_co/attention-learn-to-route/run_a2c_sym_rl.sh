for s in 1 12 123 1234
do
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --batch_size 100 --epoch_size 10000 --n_epochs 200 --graph_size 50 --no_progress_bar --val_dataset '../data/tsp/tsp50_val_seed1234.pkl' --baseline critic --distil_every 1 --il_coefficient 0 --wandb online --seed ${s}
done