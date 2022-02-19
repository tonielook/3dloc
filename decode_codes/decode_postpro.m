% clc; clear all; close all;
%% Parameters
zeta_unit = 800/10.3958;
% Np: boundary = 28, simulation = 40+28, figure4a = 100+28, colab = 176+28
Np = 100+28; zmax = 6;
px_size = [127, 117]; resolution = 10;
bol_size=[128,128,13];

% Path
% mat_path = 'C:\Users\mastaffs\Desktop\figure4a';
pred_path = 'C:\Users\DLJ\Desktop\1226_figure4a_v3';

% Save pred_label.csv & eval.csv or NOT
save_pred_info = 1;  
save_path = pred_path;
if save_pred_info
    % pred_label: pred 3d locations + flux estimated from var/dnn
    label = fopen(fullfile(save_path, 'pred_label_test.csv'), 'w');
    eval = fopen(fullfile(save_path, 'eval.csv'), 'w');
end
% Plot Figures or NOT
view = 0;

% Read Ground-truth Label and Prediction
pred = readtable(fullfile(pred_path, 'loc.csv'));
pred = table2array(pred);
pred = pred(:,2:6);
gt = readtable(fullfile(pred_path,'label.txt'));
gt = table2array(gt(:,1:5));

% gt1 = readtable(fullfile(pred_path,'DECODE_ROI3_LD_DC.csv'));
% gt1 = table2array(gt1(:,2:6));

% Initialize Evaluation Metrics
recall = zeros();
precision = zeros();
jaccard_index = zeros();
f1_score = zeros();
initial_pred_pts = zeros();
flux_all = [];

%% Post-process
tic
frame_list = unique(pred(:, 1));
for idx = 1 : 10
    nt = frame_list(idx);
    
    gt_tmp = gt(gt(:,1) == nt, :);
    pred_tmp = pred(pred(:,1) == nt, :);
    nSource = length(gt_tmp(:,1));
    
    % The same with DECODE_ROI3_LD_DC.csv
%     gt1_tmp = gt1(gt1(:,1) == nt, :);
%     gt1_tmp(:,2:3) = gt1_tmp(:,2:3)./px_size-(Np-28)/2;
%     gt1_tmp(:,4) = gt1_tmp(:,4)/zeta_unit;

    % Visualize Inital Prediction
    if view
        figure(1);
        plot3(gt_tmp(:,2),gt_tmp(:,3),gt_tmp(:,4),'ro', pred_tmp(:,2),pred_tmp(:,3),pred_tmp(:,4),'bx');
        axis([-Np/2 Np/2 -Np/2 Np/2 -zmax zmax]); grid on
        title(sprintf('Initial Prediction %d', nt))
        pause(0.5)
    end
    
    % Load Ground-truth 3d Grid
    gt_v1 = gt_tmp;
    gt_v1(:,2:3) = gt_tmp(:,2:3)+Np/2;
    gt_v1(:,4) = gt_tmp(:,4) + 7;
    
    interest_reg = zeros(32, nSource); 
    Vtrue = [gt_v1(:,2); gt_v1(:,3); gt_v1(:,4); gt_v1(:,5)];
    for i = 1 : nSource
        x0 = zeros(bol_size);
        xlow = floor(Vtrue(i)); 
        ylow = floor(Vtrue(i+nSource));
        zlow = floor(Vtrue(i+2*nSource));
        x0(xlow-1:xlow+2, ylow-1:ylow+2, zlow:zlow+1) = Vtrue(i+3*nSource);
        interest_reg(:, i) = find(x0~=0);
    end
    
    % Load Initial Prediction
    pred_v1 = pred_tmp;
    pred_v1(:,2:3) = pred_tmp(:,2:3)+Np/2;
    pred_v1(:,4) = pred_v1(:,4)+7;
    
    Vpred = [pred_v1(:,2); pred_v1(:,3); pred_v1(:,4); pred_v1(:,5)];
    pred_vol = zeros(bol_size);
    nPred = length(Vpred)/4;
    for i = 1 : nPred
        xlow = round(Vpred(i)); 
        ylow = round(Vpred(i+nPred));
        zlow = round(Vpred(i+2*nPred));
        pred_vol(xlow, ylow, zlow) = pred_vol(xlow,ylow,zlow)+Vpred(i+3*nPred);
    end

    initial_pred_pts(nt) = numel(find(pred_vol>0));
    % Visualize
    if view
        [xxtp,yytp,zztp] = ind2sub(bol_size, find(pred_vol>0));
        figure(2);
        plot3(floor(gt_v1(:,2)), floor(gt_v1(:,3)), floor(gt_v1(:,4)), 'ro', xxtp, yytp, zztp, 'bx');
        grid on
        title(sprintf('Initial Prediction to Grid %d',nt));
    end
    
    % Removing the clustered false positive 
    [xIt, elx, ely, elz] = local_3Dmax_large(pred_vol);
    idx_est = find(xIt>0);
    if isempty(idx_est)
        continue
    end
    flux_est_dnn = xIt(idx_est);
    flux_gt = gt_v1(:,5);

    %% Evaluation
    num_gt = nSource; num_pred = length(idx_est);
    [num_tr, tp_pred, tp_gt, flux_total] = evaluation(xIt, interest_reg, flux_est_dnn, flux_gt); 

    re =  num_tr/num_gt;
    pr = num_tr/num_pred; 
    ji = num_tr/(num_gt + num_pred - num_tr);
    f1 = 2*(re*pr)/(re+pr);
    
    recall(nt) = re;
    precision(nt) = pr;
    jaccard_index(nt) = ji;
    f1_score(nt) = f1;

    fprintf('Image %d\n', nt)
    fprintf('TP = %d, Pred = %d, GT = %d\n',num_tr,num_pred,num_gt);    
    fprintf('Recall = %3.2f%%, Precision = %3.2f%%\n',recall(nt)*100,precision(nt)*100);
    fprintf('---\n');

    % Shift - TP
    [xxtp, yytp, zztp] = ind2sub(bol_size, tp_pred); 
    sxtp=zeros(length(xxtp),1);  sytp=zeros(length(xxtp),1); sztp=zeros(length(xxtp),1);
    for sidx = 1: length(xxtp)
        tx = xxtp(sidx); ty = yytp(sidx); tz = zztp(sidx);
        sxtp(sidx) = elx(tx, ty, tz);
        sytp(sidx) = ely(tx, ty, tz);
        sztp(sidx) = elz(tx, ty, tz);
    end
    % Calculate Distance
    dxtp = (xxtp'+sxtp-14)*px_size(1) - (gt_tmp(tp_gt,2)+(Np-28)/2)*px_size(1);
    dytp = (yytp'+sytp-14)*px_size(2) - (gt_tmp(tp_gt,3)+(Np-28)/2)*px_size(2);
    dztp = (zztp'+sztp-7)*zeta_unit - gt_tmp(tp_gt,4)*zeta_unit;

    % Shift - FP
    [xxfp,yyfp,zzfp] = ind2sub(size(xIt), setxor(tp_pred, find(xIt>0)));
    sxfp=zeros(length(xxfp),1);  syfp=zeros(length(xxfp),1); szfp=zeros(length(xxfp),1);
    for sidx = 1: length(xxfp)
        tx = xxfp(sidx); ty = yyfp(sidx); tz = zzfp(sidx);
        sxfp(sidx) = elx(tx, ty, tz);
        syfp(sidx) = ely(tx, ty, tz);
        szfp(sidx) = elz(tx, ty, tz);
    end
    % Distance
    dxfp = ones(length(xxfp),1)*678; dyfp = ones(length(xxfp),1)*678; dzfp = ones(length(xxfp),1)*678;

    len = length(xxtp)+length(xxfp);
    xx = [xxtp';xxfp]; yy = [yytp';yyfp]; zz = [zztp';zzfp];
    sx = [sxtp;sxfp];  sy = [sytp;syfp]; sz = [sztp;szfp];
    dx = [dxtp;dxfp];  dy = [dytp;dyfp]; dz = [dztp;dzfp];
    
    %% Save pred_label.csv & eval.csv
    if save_pred_info
        LABEL = [nt*ones(len,1), xx-14, yy-14, zz, flux_total(2,:)', sx, sy, sz, dx, dy, dz];
        fprintf(label, '%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', LABEL');

        EVAL = [nt, re, pr, ji, f1];
        fprintf(eval, '%d,%.4f,%.4f,%.4f,%.4f\n', EVAL);
    end
    
    %% View
    if view
        % View Final Prediction After Post-pro - Est int
%         load(fullfile(mat_path,['I',num2str(nt),'.mat']));
        fn_gt = setxor(1:1:nSource,tp_gt);
        figure(3);
        plot3(gt_v1(tp_gt,2),gt_v1(tp_gt,3),gt_v1(tp_gt,4),'ro',...
              gt_v1(fn_gt,2),gt_v1(fn_gt,3),gt_v1(fn_gt,4),'r^',...
              xxtp,yytp,zztp,'bx',...
              xxfp,yyfp,zzfp,'b^')
        axis([0 Np-28 0 Np-28 0 zmax*2+1]); grid on;
        if isempty(xxfp)
            legend('TP-GT','TP-EST','Location','Southoutside','Orientation','horizontal')
        else
            legend('TP-GT','FN-GT','TP-EST','FP-EST','Location','Southoutside','Orientation','horizontal')
        end
        title(sprintf('Result After Postpro (Est int) %d',nt))
%         hold on; imagesc(I0); hold off
        pause(0.5)

        % View Final Prediction After Post-pro Est float
        fn_gt = setxor(1:1:nSource,tp_gt);
        figure(4);
        plot3(gt_v1(tp_gt,2),gt_v1(tp_gt,3),gt_v1(tp_gt,4),'ro',...
              gt_v1(fn_gt,2),gt_v1(fn_gt,3),gt_v1(fn_gt,4),'r^',...
              xxtp'+sxtp,yytp'+sytp,zztp'+sztp,'bx',...
              xxfp'+sxfp,yyfp'+syfp,zzfp'+szfp,'b^')
        axis([0 Np-28 0 Np-28 0 zmax*2+1]); grid on;
        if isempty(xxfp)
            legend('TP-GT','TP-EST','Location','Southoutside','Orientation','horizontal')
        else
            legend('TP-GT','FN-GT','TP-EST','FP-EST','Location','Southoutside','Orientation','horizontal')
        end
        title(sprintf('Result After Postpro (Est float) %d',nt))
    %     hold on; imagesc(I0);hold off
        pause(2)
    end
end

%% Display Mean Evaluation Metrics
mean_precision = mean(precision);
mean_recall = mean(recall);
mean_jaccard = mean(jaccard_index);
mean_f1_score = mean(f1_score);
mean_num_pts = mean(initial_pred_pts);

fprintf('Total %d Images\n', length(frame_list));
fprintf('Recall=%.2f%%, Precision=%.2f%%, Jaccard=%.2f%%, F1 socre=%.2f%%\n',...
        mean_precision*100 ,mean_recall*100, mean_jaccard*100, mean_f1_score*100);
toc

%% Save Info
if save_pred_info
    fclose(label);
    fclose(eval);
end

