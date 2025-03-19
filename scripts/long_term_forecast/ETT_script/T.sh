bash ./scripts/long_term_forecast/ETT_script/Autoformer_ETTh2_lw.sh | tee 1.txt
bash ./scripts/long_term_forecast/ETT_script/DLinear_ETTh2_lw.sh | tee 1.txt -a
bash ./scripts/long_term_forecast/ETT_script/iTransformer_ETTh2_lw.sh | tee 1.txt -a
bash ./scripts/long_term_forecast/ETT_script/Karma_ETTh2_lw.sh | tee 1.txt -a
bash ./scripts/long_term_forecast/ETT_script/S_Mamba_ETTh2_lw.sh | tee 1.txt -a