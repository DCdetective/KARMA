model_name=Mamba

seq_len=96

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 8 \
  --d_model 512 \
  --use_norm 1 \
  --use_decomp 0 \
  --embed_dim 64 \
  --des 'mambaTest' \
  --batch_size 16 \
  --learning_rate 0.00001 \
  --rec_lambda 0.99 \
  --auxi_lambda 0.01 \
  --itr 1 \

done