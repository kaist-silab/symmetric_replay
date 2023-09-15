for s in 1 12 123 1234
do
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --batch_size 100 --epoch_size 10000 --n_epochs 100 --graph_size 50 --val_dataset '../data/tsp/tsp50_val_seed1234.pkl' --seed ${s} --wandb online --baseline rollout --method gfn --k_step 2 --beta 5
done