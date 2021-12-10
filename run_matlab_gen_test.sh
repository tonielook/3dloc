set -x
# nohup sh -c '\
for pts in 5 10 15 20 25 30 35 40 45
do
/home/tonielook/MATLAB/R2021b/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "nSource = $pts;testsize=200;run('./matlab_codes/Generate_testing_images.m');exit;" 
done
> ../test_output/run_matlab.log 2>&1 &
set +x