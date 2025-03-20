model_name=Mamba

seq_len=96

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --enc_in 862 \
  --expand 1 \
  --d_state 32 \
  --d_conv 2 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --use_norm 1 \
  --use_decomp 0 \
  --learning_rate 0.001 \
  --des 'mambaTest' \
  --loss_type MSE \
  --itr 1 \

done