%  clear; close all;
% 3D localization by different method (change the variable method from 1 to 4)
% uncomment/ comment Iter_flux to do/ undo refinment on estimation of flux 
% 

load('data_natural_order_A'); % Single role 
global   Np nSource L Nzones
L = 4; Nzones = 7; 
% nSource = 5;
[Nx,Ny,Nz] = size(A); Np = Nx;
N_test  = 50;
interest_reg = zeros(32,nSource); 

time = zeros(N_test,1); 
recall = zeros(N_test,1); 
precision = zeros(N_test,1); 
ab_error = time;

time_p = zeros(N_test,1); 
recall_p = zeros(N_test,1);
precision_p = zeros(N_test,1); 
ab_error_p = ab_error;
%------------ choose different method --------------------
method = 2; % input('Enter a number from 1 to 4: '); 
% 1: KL-L1
% 2: KL-nonconvex
% 3: L2-L1
% 4: L2-nonconvex
%---------------------------------------------------

histogram_save_dir = ['../../../test_output/var/'];%,num2str(nSource),'/'];
if ~exist(histogram_save_dir, 'dir')
   mkdir(histogram_save_dir)
end
flux_all = [];
for nt = 1:N_test
    fprintf('Test %d\n',nt)
    rng(50*nt)
%% ground true and observed image not on grid point
    real_pos = zeros(nSource, 3);
    %%-------------- small region--------------------
    Flux_true = poissrnd(2000,[1,nSource]);
    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true =2*20*(rand(1,nSource)-0.5);
   
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case
    %[I0] = PointSources_poisson(nSource,Vtrue); flux = Flux_true; 
   
    % Region of interest  
    for i = 1 : nSource
        x0 = zeros(size(A));
        xlow = floor(49+Vtrue(i)); 
        ylow = floor(49+Vtrue(i+ nSource));
        zlow = floor((Vtrue(i+2*nSource)+21)/2.1)+1;
        x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*nSource); % 
        interest_reg(:,i) = find(x0~=0);
    end
    b = 5; g = poissrnd(I0+b); % Obversed image
  
%% Algorithm on localization
    tic
    switch method 
        case 1
    %% KL-L1
        Alg = 'KL-L1';
        if nSource == 15
            p1 =.01; p2 = 640; p3= 0.08;%  15 point sources case
        elseif nSource == 20  || nSource == 30
            p1 = 0.0110; p2 =  450; p3 = 0.1000;
        elseif nSource == 40
            p1 = 0.0110; p2 =  300; p3 = 0.1000;
        else
             p1 =.01; p2 = 750; p3= 0.1;
        end
        [u1] = ADMM_poisson_neg(g, A,5,p1,p2,p3); % for kl-l1 nonnegative 

        case 2
    %% IRL1-nonconvex
        Alg = 'KL-NC';
        if nSource == 30
            mu=0.01; a = 50.0000; nu = 0.0100;  lambda = 40.0000; 
        elseif nSource == 40
            mu = 0.01;a = 50.0000; nu = 0.1000; lambda = 40.0000;
        else
           mu = 0.0010;a = 80.0000; nu = 0.0400;lambda = 100.0000;
        end

        u1 = IRL1_poisson(g,A,b,a, mu,nu,lambda);


        case 3
    %% L2-L1
        Alg = 'L2-L1';
        if nSource == 15
            p1 = 0.1010; p2 =  500; p3 = 0.2000;%  15 point sources case
        elseif nSource == 20  || nSource == 30
            p1 = 0.1100; p2 =  750; p3 = 0.1200; % 20 point sources case 

        elseif nSource == 40;
            p1 = 0.0600; p2 =  600; p3 = 0.2200; 
        else
           p1 = 0.0510; p2 =  900;p3 = 0.3600; % 5, 10 point sources case
        end
        [u1] = ADMM_l2_l1(g-5, A,p1,p2,p3);    



        case 4  
    %% L2-nonconvex
        Alg = 'L2-NC';
        if nSource == 5 || nSource == 10 || nSource == 15
            mu = 0.0010;a = 20.0000; nu = 0.0700; lambda = 160.0000; %  15 point sources case
        elseif nSource == 40
             mu = 0.010;a = 20.0000; nu = 0.100; lambda = 160.0000;
        else
            mu = 0.0010;a = 20.0000; nu = 0.100;lambda = 160.0000; % 5, 10 point sources case
        end
        [u1] = IRL1_l2(g-5,A,b,a, mu,nu,lambda);    

    end
%% Removing the clustered false positive 
   time(nt) = toc;
   [xIt, elx, ely, elz] = local_3Dmax_large(u1);
    
%% Iterative Scheme on refinment on estimation of flux & Evaluation
    idx_est = find(xIt>0); 
    [flux_new] = Iter_flux(A, idx_est, g, b);

%% Evaluation 
    [re, pr,flux_total, flux_est] = Eval_v2(xIt, interest_reg,flux_new,flux); 
    % [re, pr] = Eval_v1(xIt, interest_reg); % for process without flux refinement
    recall(nt) = re;
    precision(nt) = pr;

    if recall(nt) == 0
        break;
    end

    flux_all = [flux_all,flux_total];

%     figure(1); % Plot flux estimation
%     w1 = 0.5; w2 = 0.25;
%     bar(flux_est(1,:),'FaceColor',[0.2 0.2 0.5])
%     hold on 
%     bar(flux_est(2,:),w2,'FaceColor',[0 0.7 0.7])
%     legend('true','est')
%     title(num2str(nt));
%     set(gcf,'position',[100,100,1200,600]);
%     saveas(gcf,[histogram_save_dir,'/',num2str(nt),'.fig']);
%     saveas(gcf,[histogram_save_dir,'/',num2str(nt),'.png']);
%     hold off
    
    fprintf('%s in %d point source case\n',Alg,nSource);
    fprintf('Recall = %3.2f%%\n',recall(nt)*100);
    fprintf('Precision = %3.2f%%\n',precision(nt)*100);
    fprintf('Cost time = %3.2f seconds\n',time(nt));
    fprintf('---\n');
 
end

mean_recall=mean(recall);
mean_precision=mean(precision);

dlmwrite([histogram_save_dir,'result.csv'],{N_test,nSource,mean_precision,mean_recall},'delimiter',',','-append');

%% plot the 3D estimation (compare ground true with estimated solution)
% [loc_x,loc_y,loc_z] = ind2sub(size(A),find(xIt>0)); 
% Ax_eta_2d=ifftn(fftn(A).*fftn(ifftshift(fftshift(xIt),3))); 
% J = Ax_eta_2d(:,:,Nz);
% figure;
% scatter3(49+Vtrue(1:nSource),49+Vtrue(nSource+1:2*nSource), ...
%     (Vtrue(2*nSource+1:3*nSource)+21)/2.1+1,'ro')
% axis([1 96 1 96 0 21])
% hold on
% scatter3(loc_x,loc_y,loc_z,'b+')
% % scatter3(49+predy,49+predx,(predz+21)/2.1+1,'bx')
% legend('true','est','est-nn','Location','Southoutside','Orientation','horizontal')
% imagesc(I0.')
% % view([90, 90])

%% Histogram on flux estimation 
% figure; % plot the estimated and true value of flux  wrt histogram
% histogram(flux_total(1,:))
% hold on
% histogram(flux_total(2,:))
% hold off 
% legend('true','est') 
% flux_per = abs(flux_all(1,:)-flux_all(2,:))./flux_all(1,:);
% figure; % plot the relative error in flux wrt histogram 
% histogram(flux_per)
% title(num2str(nSource));
