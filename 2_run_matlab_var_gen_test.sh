set -x
# nohup sh -c '\

echo "Begin:" `date` >> ../test_output/result_var.csv

# for pts in 40 35 30 25 20 15 10 5 45
for pts in 5 10 15 20 25 30 35 40 45
do
# /home/tonielook/MATLAB/R2021b/bin/matlab \
#     -nodisplay -nosplash -nodesktop \
#     -r "nSource = $pts;run('./matlab_codes/demo_rpsf/demo.m');exit;" 
/home/tonielook/MATLAB/R2021b/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "nSource = $pts;run('./matlab_codes/demo_rpsf/postpro_loc_v3.m');exit;" 
done 
> ../test_output/run_matlab.log 2>&1 &
echo "End:" `date` >> ../test_output/result_var.csv
