model_name=DLinear
for seq_len in 96 192 336 720 1024 1440 2048; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$seq_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $seq_len \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1 \
  --batch_size 16

done