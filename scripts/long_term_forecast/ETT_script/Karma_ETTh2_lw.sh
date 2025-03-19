model_name=Karma

pred_len=96
# using haar for 192, 336
for seq_len in 48 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 256 \
  --use_norm 1 \
  --use_decomp 0 \
  --embed_dim 64 \
  --des 'Karma' \
  --itr 1 \

done