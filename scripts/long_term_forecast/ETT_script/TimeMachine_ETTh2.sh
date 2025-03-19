model_name=TimeMachine

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2024
one=96
two=192
three=336
four=720
residual=1
fc_drop=0.7
dstate=256
dconv=2
for seq_len in 96
do
    for pred_len in 96 192 336 720
    do
        for e_fact in 1
        do

            if [ $pred_len -eq $one ]
            then
                n1=128
                n2=32
            fi
            if [ $pred_len -eq $two ]
            then
                n1=256
                n2=32
            fi
            if [ $pred_len -eq $three ]
            then
                n1=512
                n2=64
            fi
            if [ $pred_len -eq $four ]
            then
                n1=256
                n2=128
            fi
            python -u run.py \
            --use_multi_gpu \
            --task_name long_term_forecast \
            --random_seed $random_seed \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id $model_id_name$seq_len'_'$pred_len \
            --model $model_name \
            --data $data_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --n1 $n1 \
            --n2 $n2 \
            --dropout $fc_drop\
            --use_norm 1 \
            --norm_method 'RevIN' \
            --ch_ind 1 \
            --residual $residual\
            --d_conv $dconv \
            --d_state $dstate\
            --expand $e_fact\
            --des 'Exp' \
            --itr 1 \
            --batch_size 512 \
            --learning_rate 0.001

        done
    done
done