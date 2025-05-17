nohup python train.py \
  --project cnn_project \
  --batch_size 128 \
  --max_epochs 50 \
  > train.log 2>&1 &