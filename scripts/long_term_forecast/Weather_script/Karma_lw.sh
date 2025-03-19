model_name=Karma

pred_len=96

for seq_len in 48 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --enc_in 21 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 21 \
  --d_model 128 \
  --use_norm 1 \
  --use_decomp 1 \
  --embed_dim 128 \
  --des 'Karma' \
  --itr 1 \

done