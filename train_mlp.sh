nohup python train.py \
  --project my_mlp_project \
  --batch_size 128 \
  --max_epochs 50 \
  > train.log 2>&1 &