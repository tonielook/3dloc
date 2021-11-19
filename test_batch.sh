#!/bin/bash
set -x
# declare -a cps=("1104-nTrain2499-lr0.0005-Epoch190-batchSize8-D250-cnn_residual/checkpoint_best_loss" 
#                 "1116-nTrain2500-lr0.0005-Epoch190-batchSize8-D250-cnn_residual/checkpoint_best_loss" 
# )
# pts=(5 25 45)

declare -a cps=("1119-nTrain6000-lr0.0005-Epoch190-batchSize8-D250-cnn_residual/checkpoint_best_loss")
pts=(5 10 15 20 25 30 35 40 45)

for cp in "${cps[@]}"; do
    echo "$cp" >> ../test_output/postpro_result.csv
    for pt in ${pts[*]}; do
        python3 main.py      --gpu_number='1' \
            --checkpoint_path='../../trained_model/'$cp \
            --training_data_path='../data_test/test'$pt \
            --result_path='../test_output/test'$pt \
            --model_use='cnn_residual'  \
            --post_pro=0  \
            --D=250  \
            --zmax=20  \
            --clear_dist=1  \
            --upsampling_factor=2  \
            --scaling_factor=170  \
            --batch_size=8  \
            --train_or_test='test' 
                    
        /home/tonielook/MATLAB/R2021b/bin/matlab -nodisplay -nosplash -nodesktop \
            -r "nSource = $pt;run('./matlab_codes/postpro_loc_batch.m');exit;" 
    done
done
set +x


# /home/lingjia/Documents/3dloc_result/CNN/0808-nTrain9000-lr0.0007-Epoch100-batchSize8-D250-cnn_residual/checkpoint_best_loss
# /home/lingjia/Documents/3dloc_result/CNN_v2/0909-nTrain9000-lr0.0005-Epoch300-batchSize8-D250-cnn_residual/checkpoint_best_loss