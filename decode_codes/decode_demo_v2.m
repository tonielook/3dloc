%% Generate images
% clear; close all

%% Information of Inference Dataset
% /media/hdd/3dloc_data/DECODE_Setting/figure4a/DECODE_ROI3_LD_DC_px.csv
% /media/hdd/3dloc_data/DECODE_Setting/simulation/label_colab.csv
path = '/media/hdd/3dloc_data/DECODE_Setting/figure4a/DECODE_ROI3_LD_DC.csv';
gt_label = readtable(path);
gt_label = table2array(gt_label); % [frame_id, emitter_id, x, y, z, photon]

for i = 1:6
    fprintf('min = %.2f, max = %.2f\n',min(gt_label(:,i)),max(gt_label(:,i)));
end
nSource_dist_v1 = tabulate(gt_label(:,2));
nSource_dist = tabulate(nSource_dist_v1(:,2));
fprintf('min = %d, max = %d, mean = %.2f\n',min(nSource_dist_v1(:,2)),max(nSource_dist_v1(:,2)),mean(nSource_dist_v1(:,2)))
% figure;plot(nSource_dist(:,1),nSource_dist(:,2)); title('Point Source Density')

% Photon
% histogram(gt_label(:, 6));

% Flux
p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
zeta_unit = 800/10.3958;
gt_zeta = gt_label(:, 5)/zeta_unit;
pred_ratio = p(1)*abs(gt_zeta').^4 + p(2)*abs(gt_zeta').^3 + p(3)*abs(gt_zeta').^2 + p(4)*abs(gt_zeta') + p(5);
flux = gt_label(:, 6)./pred_ratio';
% histogram(flux);

% flux_extent = [3400, 3.5e4];
flux_extent = [70, 170];


%% Parameter Setting
global   Np L Nzones nSource
L = 4;  Nzones = 7; Np = 100+28;
% Np: boundary = 28, simulation = 40+28, figure4a = 100+28, colab = 176+28

zeta_unit = 800/10.3958;
Ntest = [1, 10000];

% Image Extent
% figure4a  [0,83],[0,96], [-400,400] -> [100,100,400]
% figure_colab [176,176,1194]
xextent = 100; yextent = 100; zextent = ceil(400/zeta_unit)*2;

all_photon = [];
all_flux = [];
all_depth = [];
all_nSource = [];

% figure4a LD=10
rng(1024);
nSources = randi ([5,23], 1, 10000);

save_path = '/media/hdd/3dloc_data/DECODE/simulation_figure4a_LD';
if ~exist(save_path, 'dir')
   mkdir(fullfile(save_path, 'clean'))
   mkdir(fullfile(save_path, 'noise'))
   mkdir(fullfile(save_path, 'clean_img'))
   mkdir(fullfile(save_path, 'noise_img'))
   fprintf(['Save path ', save_path, ' does not exist, create now\n'])
end

label_file = fopen(fullfile(save_path,'label.txt'),'w');
%% generate images
for frame_id = Ntest(1) : Ntest(2)
    nSource = nSources(frame_id);
    
    rng(frame_id);
    Xp_true = xextent * (rand(1,nSource) - 0.5);
    Yp_true = yextent * (rand(1,nSource) - 0.5);
    zeta_true = zextent * (rand(1,nSource) - 0.5);
    
    p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    Flux_true = (rand(1, nSource) - 0.5) * diff(flux_extent) + mean(flux_extent);
    Flux_true = pred_ratio'.*Flux_true;

    % generate image based on 3d location and photon values
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource, Vtrue); % flux value in normalized basis case
    % figure; imshow(I0,[]); hold on; plot(Yp_true+Np/2, Xp_true+Np/2, 'ro')

    %%% Background %%%
    % bg_range = [20,200];
    % bg_param = [(bg_range(2)-bg_range(1))/2, (bg_range(2)+bg_range(1))/2]; %[scale, ratio]
    % bg = (rand(1)-0.5)*bg_param(1) + bg_param(2);
    % p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    % pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    % bg = pred_ratio'.*bg;


    %%% Noise - parameters from DECODE %%%
%     bg = 5;
%     g = I0 + bg;
%     qe = 1.0;
%     spur = 0.0015;
%     em_gain = 100.0;
%     read_sigma = 58.8;
%     g = poissrnd(g*qe+spur);
%     g = gamrnd(g,em_gain);
%     g = g + normrnd(0,read_sigma);

    bg = 5;
    g = poissrnd(I0+bg);
    % figure; imshow(g, []); hold on; plot(Yp_true+Np/2, Xp_true+Np/2,'ro')

    % save mat file
    all_photon = [all_photon; Flux_true'];
    all_flux = [all_flux; flux];
    all_depth = [all_depth; zeta_true'];
    all_nSource = [all_nSource, nSource];

    save(fullfile(save_path, 'noise', ['im', num2str(frame_id), '.mat']), 'g');
    save(fullfile(save_path, 'clean', ['im', num2str(frame_id), '.mat']), 'I0');
    
    I0 = mat2gray(I0);
    g = mat2gray(g);
    imwrite(g, fullfile(save_path, 'noise_img', ['im', num2str(frame_id), '.png']));
    imwrite(I0, fullfile(save_path, 'clean_img', ['im', num2str(frame_id), '.png']));
    fprintf([num2str(frame_id), ' saved\n']);
    
    % save labels
     LABEL = [frame_id * ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
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