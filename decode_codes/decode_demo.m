%% Generate images
% clear; close all
global   Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L

zeta_unit = 800/10.3958;
zmax = ceil(400/zeta_unit);
% Np: boundary = 28, simulation = 40+28, figure4a = 100+28, colab = 176+28
Np = 100+28;
half_xy = (Np-28)/2;
px_size = [127, 117];

%%% Image Extent %%%
% figure4a  [0,83],[0,96], [-400,400] -> [100,100,400]
% figure_colab [176,176,1194]

%% Load label.csv
% /media/hdd/3dloc_data/DECODE_Setting/figure4a/DECODE_ROI3_LD_DC_px.csv
% /media/hdd/3dloc_data/DECODE_Setting/figure_colab/label_v1.csv
path = '/media/hdd/3dloc_data/DECODE_Setting/figure4a/DECODE_ROI3_LD_DC.csv';

gt_label = readtable(path);
gt_label = table2array(gt_label); % [frame_id, emitter_id, x, y, z, photon]
for i = 1:6
    fprintf('min = %.2f, max = %.2f\n',min(gt_label(:,i)),max(gt_label(:,i)));
end

all_photon = [];
all_flux = [];
all_depth = [];
all_nSource = [];

save_path = '/media/hdd/3dloc_data/DECODE/figure4a_v3';
if ~exist(save_path, 'dir')
   mkdir(fullfile(save_path, 'clean'))
   mkdir(fullfile(save_path, 'noise'))
   mkdir(fullfile(save_path, 'clean_img'))
   mkdir(fullfile(save_path, 'noise_img'))
   fprintf(['Save path ', save_path, ' does not exist, create now\n'])
end

label_file = fopen(fullfile(save_path,'label.txt'),'w');
frame_ids = unique(gt_label(:, 2));
%% generate images
for idx = 1 : length(frame_ids)
    frame_id = frame_ids(idx);
    tmp_label = gt_label(gt_label(:, 2) == frame_id, :);
%     tmp_label = tmp_label (tmp_label(:, 7)>0.6, [1,2,3,4,5,6]);
    
    Yp_true = tmp_label(:, 3)/px_size(1) - half_xy;
    Xp_true = tmp_label(:, 4)/px_size(2) - half_xy;
    zeta_true = tmp_label(:, 5)/zeta_unit;
    
    % Visualize Ground-truth Image
%     figure; scatter3(Xp_true, Yp_true, zeta_true, 'ro'); 
%     axis([-Np/2 Np/2 -Np/2 Np/2 -zmax zmax])
    
    nSource = length(Xp_true);
    photon = ones(nSource, 1)*2000;
    
    % figure41=./50
%     photon = tmp_label(:,6)/4

    % generate image based on 3d location and photon values
    vtrue = [Xp_true; Yp_true; zeta_true; photon];
    [I0,flux] = PointSources_poisson_v2(nSource,vtrue); % flux value in normalized basis case
    % figure; imshow(I0,[]); hold on; plot(Yp_true+Np/2, Xp_true+Np/2, 'ro');

    %%% Background %%%
    % bg_range = [20,200];
    % bg_param = [(bg_range(2)-bg_range(1))/2, (bg_range(2)+bg_range(1))/2]; %[scale, ratio]
    % bg = (rand(1)-0.5)*bg_param(1) + bg_param(2);
    % p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    % pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    % bg = pred_ratio'.*bg;
%     bg = 5;
%     g = I0 + bg;

    %%% Noise - parameters from DECODE %%%
%     qe = 1.0;
%     spur = 0.0015;
%     em_gain = 100.0;
%     read_sigma = 58.8;
%     g = poissrnd(g*qe+spur);
%     g = gamrnd(g,em_gain);
%     g = g + normrnd(0,read_sigma);

    bg = 5;
    g = poissrnd(I0+bg);
    % figure; imshow(g, []); hold on; plot(Yp_true+Np/2, Xp_true+Np/2, 'ro')

    % save mat file
    all_photon = [all_photon; photon];
    all_flux = [all_flux; flux];
    all_depth = [all_depth; zeta_true];
    all_nSource = [all_nSource, nSource];

    save(fullfile(save_path, 'noise', ['im', num2str(frame_id), '.mat']), 'g');
    save(fullfile(save_path, 'clean', ['im', num2str(frame_id), '.mat']), 'I0');
    
    I0 = mat2gray(I0);
    g = mat2gray(g);
    imwrite(g, fullfile(save_path, 'noise_img', ['im', num2str(frame_id), '.png']));
    imwrite(I0, fullfile(save_path, 'clean_img', ['im', num2str(frame_id), '.png']));
    fprintf([num2str(frame_id), ' saved\n']);
    
    % Save Labels
    LABEL = [frame_id*ones(1,nSource); Yp_true'; Xp_true'; zeta_true'; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);
end
fclose(label_file);


%% save currnt figure as png
% filename = 'C:\Users\DLJ\Desktop\noise.png';
% frame = getframe(gca); 
% img = frame2im(frame); 
% imwrite(img,filename); 

%% save all_flux, all_nSource, all_nSource_v1
% save(fullfile(save_path,'photons.mat'),'all_photon');
% save(fullfile(save_path,'flux.mat'),'all_flux');
% save(fullfile(save_path,'nSource.mat'),'all_nSource');
% save(fullfile(save_path,'depth.mat'),'all_depth');

%% visualize flux, all_nSource, all_nSource_v1
% figure; histogram(all_photon); title('Flux: photon');
% figure; histogram(all_flux); title('flux');
% figure; histogram(all_depth); title('depth');
% figure; histogram(all_nSource); title('nSource');

%% visualize 3d plot
% figure;
% plot3(Xp_true,Yp_true,zeta_true,'o');
% xlabel('y')
% ylabel('x')
% zlabel('zeta')
% xlim([-48 48])
% ylim([-48 48])
% zlim([-20 20])
% ax = gca;
% ax.YDir = 'reverse';
% grid on