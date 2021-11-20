set -x
nohup python3 main.py           --gpu_number='2,3'  \
                                --zmax=20  \
                                --D=250  \
                                --clear_dist=1  \
                                --upsampling_factor=2  \
                                --scaling_factor=170  \
                                --training_volume=2000 \
                                --batch_size=8  \
                                --initial_learning_rate=0.0005  \
                                --lr_decay_per_epoch=3  \
                                --lr_decay_factor=0.5  \
                                --saveEpoch=10  \
                                --maxEpoch=190  \
                                --train_or_test='train'  \
                                --training_data_path='../data_train'  \
                                --result_path='../../trained_model'  \
                                --resume_training=1  \
                                --checkpoint_path='../../trained_model/1119-nTrain3000-lr0.0005-Epoch190-batchSize8-D250-cnn_residual-resume/checkpoint_best_loss'  \
                                --model_use='cnn_residual'  \
                                > ../../trained_model/training_main_sh.log 2>&1 &
set +x