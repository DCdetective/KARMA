model_name=Karma

seq_len=96

for pred_len in 96 192
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_state 2 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 256 \
  --use_norm 1 \
  --use_decomp 0 \
  --embed_dim 32 \
  --des 'Karma' \
  --itr 1 \

done

for pred_len in 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_state 2 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 512 \
  --use_norm 1 \
  --use_decomp 0 \
  --embed_dim 32 \
  --des 'Karma' \
  --itr 1 \

done