model_name=Mamba

seq_len=96

for pred_len in 96 192 336 720
# for pred_len in 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --enc_in 321 \
  --expand 1 \
  --d_state 16 \
  --d_conv 2 \
  --c_out 321 \
  --des 'mambaTest' \
  --d_model 512 \
  --batch_size 16 \
  --use_norm 1 \
  --use_decomp 0 \
  --learning_rate 0.0005 \
  --itr 1 \

done
