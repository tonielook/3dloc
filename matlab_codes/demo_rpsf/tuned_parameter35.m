clear; close all
% 3D localization by different method (change the variable method from 1 to 4)
% uncomment/ comment Iter_flux to do/ undo refinment on estimation of flux 
% 
addpath('../input_data/');
addpath(genpath('../common_func'));

load('data_natural_order_A'); % Single role 
global   Np nSource L Nzones
L = 4; Nzones = 7; nSource = 35;
[Nx,Ny,Nz] = size(A); Np = Nx;
N_test  = 20;
interest_reg = zeros(32,nSource); 
time = zeros(N_test,1); recall = zeros(N_test,1); 
precision = zeros(N_test,1); 
res = [];
%================ choose different method =====================
method = 2; % input('Enter a number from 1 to 4: '); 
% 1: KL-L1
% 2: KL-nonconvex
% 3: L2-L1
% 4: L2-nonconvex
%==============================================================

%================ whether with or without flux refinement ========
Flux_ref = 0; 
% 1 is with
% 0 is without
%==================================================================
fileID = fopen('35KL-NC_step.txt','a');
% for p1 =  0.01 % 0.001:0.003:0.008
%     for p2 = 750 % 300:150:900
%         for p3 = 0.1 % 0.02:0.08:0.4

a = 50.0000;
for mu = 0.0045:0.0001:0.0055
for nu = 0.025:0.001:0.025
for lambda = 40:40:40
for nt = 1: N_test % flux test using 38
    fprintf('Test %d\n',nt)
    rng(100*nt)
%% ground true and observed image not on grid point
    real_pos = zeros(nSource, 3);
    %%-------------- small region--------------------
    Flux_true = poissrnd(2000,[1,nSource]);
    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true =2*20*(rand(1,nSource)-0.5);
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case   
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
        [u1] = ADMM_poisson_neg_modified(g, A,5,p1,p2,p3); % for kl-l1 nonnegative 

        case 2
    %% IRL1-nonconvex
        Alg = 'KL-NC';
%         if nSource == 30
%             mu=0.01; a = 50.0000; nu = 0.0100;  lambda = 40.0000; 
%         elseif nSource == 40
%             mu = 0.01;a = 50.0000; nu = 0.1000; lambda = 40.0000;
%         else
%            mu = 0.0010;a = 80.0000; nu = 0.0400;lambda = 100.0000;
%         end

%         u1 = IRL1_poisson_modified(g,A,b,a, mu,nu,lambda);
        u1 = IRL1_poisson_steplength(g,A,b,a, mu,nu,lambda);


        case 3
    %% L2-L1
        Alg = 'L2-L1';
        if nSource == 15
            p1 = 0.1010; p2 =  500; p3 = 0.2000;%  15 point sources case
        elseif nSource == 20  || nSource == 30
            p1 = 0.1100; p2 =  750; p3 = 0.1200; % 20 point sources case 

        elseif nSource == 40
            p1 = 0.0600; p2 =  600; p3 = 0.2200; 
        else
           p1 = 0.0510; p2 =  900;p3 = 0.3600; % 5, 10 point sources case
        end
        [u1] = ADMM_l2_l1(g-5, A,p1,p2,p3);    
        case 4  
    %% L2-nonconvex
        Alg = 'L2-NC';
%         if nSource == 5 || nSource == 10 || nSource == 15
%             mu = 0.0010;a = 20.0000; nu = 0.0700; lambda = 160.0000; %  15 point sources case
%         elseif nSource == 40
%              mu = 0.010;a = 20.0000; nu = 0.100; lambda = 160.0000;
%         else
%             mu = 0.0010;a = 20.0000; nu = 0.100;lambda = 160.0000; % 5, 10 point sources case
%         end
        [u1] = IRL1_l2_steplength(g-5,A,b,a, mu,nu,lambda);    

    end
%% Removing the clustered false positive 
   time(nt) = toc;
   [xIt, elx, ely, elz] = local_3Dmax_large(u1);
    
%% Iterative Scheme on refinment on estimation of flux & Evaluation
    if Flux_ref == 1
        idx_est = find(xIt>0); 
        [flux_new] = Iter_flux(A, idx_est, g, b);
        [re, pr,flux_total, flux_est] = ...
            Eval_v2(xIt, interest_reg,flux_new,flux); 
    else 
        [re, pr] = Eval_v1(xIt, interest_reg); % without flux refinement
    end
    
    recall(nt) = re;
    precision(nt) = pr;

    if recall(nt) == 0
        break;
    end

    fprintf('%s in %d point source case\n',Alg,nSource);
    fprintf('Recall = %3.2f%%\n',recall(nt)*100);
    fprintf('Precision = %3.2f%%\n',precision(nt)*100);
    fprintf('Cost time = %3.2f seconds\n',time(nt));
    fprintf('---\n');
 
end
fprintf(fileID,'\n==========\n');
% fprintf(fileID,'p1 = %5.4f; p2 = %4d; p3 = %5.4f; \n',p1, p2, p3);
 fprintf(fileID,'mu = %5.4f;a = %5.4f; nu = %5.4f; lambda = %5.4f; \n',mu,a,nu,lambda);
fprintf(fileID,'Recall rate = %3.2f%% \n',mean(recall)*100);
fprintf(fileID,'Prec = %3.2f%% \n',mean(precision)*100);
fprintf(fileID,'average = %3.2f%% \n',(mean(precision)+mean(recall))*50);
tem_res = [a mu nu lambda mean(recall) mean(precision) (mean(precision)+mean(recall))/2];
res = [res;tem_res];
end
end
end
 [val, idx] = max(res(:,7));
 a = res(idx,1); mu = res(idx,2); nu = res(idx,3); lambda = res(idx,4);
N_test = 50;
recall = zeros(N_test,1); 
precision = zeros(N_test,1); 
recall_p = zeros(N_test,1);
precision_p = zeros(N_test,1);

 for nt = 1: N_test % flux test using 38
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
%  [u1] = IRL1_l2_steplength(g-5,A,b,a, mu,nu,lambda);    
%% Removing the clustered false positive 
     [xIt, elx, ely, elz] = local_3Dmax_large(u1);
     [re, pr] = Eval_v1(xIt, interest_reg);
     recall(nt) = re;
     precision(nt) = pr;
     [re_p, pr_p] = Eval_v1(u1, interest_reg); 
     recall_p(nt) = re_p;
    precision_p(nt) = pr_p;
 end
 
save('result_35KL-NC_step.mat','a','mu','nu','lambda',...
    'recall','precision','recall_p','precision_p');
