%% based on label.txt and 3d grid(250*384*384) plot figure on 96*96*41
result_path = 'D:\OneDrive - City University of Hong Kong\Microscope\test_result\pts_5wc_v1\model22_latest_v0';
pred = readtable([result_path,'\loc_bool.csv']);
pred = table2array(pred);
gt = readtable([result_path,'\label.txt']);
gt = table2array(gt(:,1:5));

W = 96;
H = 96;
D = 250;
zmax = 20;
zmin = -zmax;
upsampling_factor = 4;
pixel_size_axial = (zmax - zmin + 2*1)/D;

id=1;
pred_tmp = pred(pred(:,1)==id,:);
gt_tmp = gt(gt(:,1)==id,:);

% post-processing
pred_bol = zeros(96*4,96*4,250);
for ii = 1:length(pred_tmp)
    pred_bol(pred_tmp(ii,2)+1,pred_tmp(ii,3)+1,pred_tmp(ii,4)+1) = pred_tmp(ii,5);
end
[xIt, elx, ely, elz] = local_3Dmax_large(pred_bol);  % for mat from python [z x y]

[re, pr,flux_total, flux_est] = Eval_v2(xIt, interest_reg,flux_new,flux);

px = [];
py = [];
pz = [];
xIt_p = find(xIt>0);
for n = 1 : numel(xIt_p)
    [p_x, p_y, p_z] = ind2sub(size(xIt),xIt_p(n));
    p_x1 = p_x + elx(p_x,p_y,p_z);
    p_y1 = p_y + ely(p_x,p_y,p_z);
    p_z1 = p_z + elz(p_x,p_y,p_z);
    px = [px p_x1];
    py = [py p_y1];
    pz = [pz p_z1];
end
predx = px/upsampling_factor - W/2;
predy = py/upsampling_factor - H/2;
predz = pz*pixel_size_axial+zmin;






figure;
plot3(gt_tmp(:,2),gt_tmp(:,3),gt_tmp(:,4),'o',predx,predy,predz,'x');
xlabel('y 3')
ylabel('x 2')
zlabel('zeta 1')
xlim([-48 48])
ylim([-48 48])
zlim([-20 20])
title(['img',num2str(id)])
ax = gca;
ax.YDir = 'reverse';
grid on



% % plot figure without post-processing
% result_path = 'D:\OneDrive - City University of Hong Kong\Microscope\test_result\pts_5wc_v1\model22_latest_v0';
% pred_v1 = readtable([result_path,'\loc.csv']);
% pred_v1 = table2array(pred_v1);
% pred_tmp_v1 = pred_v1(pred_v1(:,1)==id,:);
% 
% figure;
% plot3(gt_tmp(:,2),gt_tmp(:,3),gt_tmp(:,4),'o',pred_tmp_v1(:,2),pred_tmp_v1(:,3),pred_tmp_v1(:,4),'x');
% xlabel('y 3')
% ylabel('x 2')
% zlabel('zeta 1')
% title(['img',num2str(id)])
% ax = gca;
% ax.YDir = 'reverse';
% grid on



