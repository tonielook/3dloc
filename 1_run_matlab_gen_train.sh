set -x
/home/tonielook/MATLAB/R2021b/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "run('./matlab_codes/Generate_training_images.m');exit;" 
done
> ../test_output/run_matlab_gen_train.log 2>&1 &
set +x