%% Parameters - demo.py (Generating Data)
load('data_natural_order_A'); % Single role
global Np nSource L Nzones
L = 4; Nzones = 7; b = 5; [Nx,Ny,Nz] = size(A); Np = Nx;
nSource = 10; zmax = 20;

% Read Ground-truth Label and Prediction
mat_path = 'F:\Tmp\0118_Evalution_with_Float\testset';
mat_path = fullfile(mat_path,['test',num2str(nSource)]);
pred_path = 'F:\Tmp\0118_Evalution_with_Float\HS';
pred_path = fullfile(pred_path,['test',num2str(nSource)]);

pred = readtable(fullfile(pred_path,'loc.csv'));
pred = table2array(pred(:,2:6));
% pred = table2array(pred);
% Due to Problem in Post-pro of CNN, xy-0.5/2=0.25, z-0.5*0.172=0.086
% pred(:,2:3) = pred(:,2:3)-0.25;
% pred(:,4) = pred(:,4)-0.086;

gt = readtable(fullfile(pred_path,'label.txt'));
gt = table2array(gt(:,1:5));

% Initialize Evaluation Metrics
recall = zeros();
precision = zeros();
jaccard_index = zeros();
f1_score = zeros();
initial_pred_pts = zeros();
flux_all = [];

% Save pred_label.csv & eval.csv or NOT
view = 1;
save_pred_info = 0;
save_path = pred_path;
if save_pred_info
    label = fopen(fullfile(save_path, 'pred_label.csv'), 'w');
    eval = fopen(fullfile(save_path, 'eval_v1.csv'), 'w');
end

%% Post-pro
tic
for nt = 56
    gt_tmp = gt(gt(:,1)==nt,:);
    pred_tmp = pred(pred(:,1)==nt,:);

    if view
        % View Initial Prediction
        figure(1)
        plot3(gt_tmp(:,2),gt_tmp(:,3),gt_tmp(:,4),'ro', pred_tmp(:,2),pred_tmp(:,3),pred_tmp(:,4),'bx');
        title(sprintf('Initial Prediction %d', nt))
        axis([-Np/2 Np/2 -Np/2 Np/2 -zmax zmax]); grid on
%         pause(0.5)
    end

    % Load Ground Truth 3D Grid
    interest_reg = zeros(32,nSource); 
    Vtrue = [gt_tmp(:,2);gt_tmp(:,3);gt_tmp(:,4);gt_tmp(:,5)];
    flux_gt = gt_tmp(:,5);
    for i = 1 : nSource
        x0 = zeros(size(A));
        xlow = floor(49+Vtrue(i));
        ylow = floor(49+Vtrue(i+nSource));
        zlow = floor((Vtrue(i+2*nSource)+21)/2.1)+1;
        x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*nSource);
        interest_reg(:,i) = find(x0~=0);
    end
    
    % Load Initial Prediction
    Vpred = [pred_tmp(:,2);pred_tmp(:,3);pred_tmp(:,4);pred_tmp(:,5)];
    pred_vol = zeros(size(A));
    nPred = length(Vpred)/4;
    for i = 1 : nPred
        xlow = round(49+Vpred(i)); 
        ylow = round(49+Vpred(i+nPred));
        zlow = round((Vpred(i+2*nPred)+21)/2.1)+1;
        pred_vol(xlow,ylow,zlow)= pred_vol(xlow,ylow,zlow)+Vpred(i+3*nPred);
    end

    initial_pred_pts(nt) = numel(find(pred_vol>0));
    if view
        % View Initial Prediction to Grid
        figure(2)
        [xx,yy,zz] = ind2sub(size(A), find(pred_vol>0));
        plot3(floor(gt_tmp(:,2)+49), floor(gt_tmp(:,3)+49), floor((gt_tmp(:,4)+21)/2.1+1), 'ro', xx, yy, zz, 'bx');
        title(sprintf('Initial Prediction to Grid %d',nt));
        axis([0 Np+1 0 Np+1 0 21]); grid on;
%         pause(0.5)
    end
    
    % Removing Clustered False Positive 
    [xIt, elx, ely, elz] = local_3Dmax_large(pred_vol);
    
    idx_est = find(xIt>0); 
    if isempty(idx_est)
        continue
    end
    
    flux_est_dnn = xIt(idx_est);
    % Refinment on Estimation of Flux
%     load(fullfile(mat_path,['im',num2str(nt),'.mat']));  % mat file for g
%     flux_est_var = Iter_flux(A, idx_est, g, b);

    %% Evaluation
    num_gt = nSource; num_pred = length(idx_est);
    [num_tr,tp_pred,tp_gt,flux_total] = evaluation(xIt, interest_reg, flux_est_dnn, flux_gt);
%     [num_tr,tp_pred,tp_gt,flux_total] = evaluation_v2(xIt, interest_reg, flux_est_dnn, flux_gt,nSource);
%     [num_tr,tp_pred,tp_gt,flux_total] = evaluation_v3(gt_tmp(:,2:5),xIt,elx,ely,elz,flux_est_dnn);

    re = num_tr/num_gt;
    pr = num_tr/num_pred; 
    ji = num_tr/(num_gt + num_pred - num_tr);
    f1 = 2*(re*pr)/(re+pr);
    
    recall(nt) = re;
    precision(nt) = pr;
    jaccard_index(nt) = ji;
    f1_score(nt) = f1;
    
    fprintf('Image %d in %d point source case\n', nt,nSource)
    fprintf('TP = %d, Pred = %d, GT = %d\n',num_tr,num_pred,num_gt);    
    fprintf('Recall = %3.2f%%, Precision = %3.2f%%\n',recall(nt)*100,precision(nt)*100);
    fprintf('---\n');
    
    %% Save Results
    % TP
    [xxtp,yytp,zztp] = ind2sub(size(A), tp_pred); 
    sxtp = zeros(length(xxtp),1);  sytp = zeros(length(xxtp),1); sztp = zeros(length(xxtp),1);
    for sidx = 1: length(xxtp)
        tx = xxtp(sidx); ty = yytp(sidx); tz = zztp(sidx);
        sxtp(sidx) = elx(tx, ty, tz);
        sytp(sidx) = ely(tx, ty, tz);
        sztp(sidx) = elz(tx, ty, tz);
    end

    % FP
    [xxfp,yyfp,zzfp] = ind2sub(size(xIt), setxor(tp_pred, find(xIt>0)));
    sxfp = zeros(length(xxfp), 1);  syfp = zeros(length(xxfp), 1); szfp = zeros(length(xxfp), 1);
    for sidx = 1: length(xxfp)
        tx = xxfp(sidx); ty = yyfp(sidx); tz = zzfp(sidx);
        sxfp(sidx) = elx(tx, ty, tz);
        syfp(sidx) = ely(tx, ty, tz);
        szfp(sidx) = elz(tx, ty, tz);
    end

    xx=[xxtp';xxfp]; yy=[yytp';yyfp]; zz=[zztp';zzfp];
    sx=[sxtp;sxfp]; sy=[sytp;syfp]; sz=[sztp;szfp];
    
    %% Save pred_label.csv & eval.csv
    if save_pred_info
        LABEL = [nt*ones(1,length(xx))', xx, yy, zz, flux_total(2,:)', sx, sy, sz];
        fprintf(label, '%d,%.4f,%.4f,%.4f,%.4f\n', LABEL);
        
        EVAL = [nt, re, pr, ji, f1];
        fprintf(eval, '%d,%.4f,%.4f,%.4f,%.4f\n', EVAL);
    end
    
    %% View
    if view
        % View Final Prediction After Post-pro - Est int
%         load(fullfile(mat_path,['I',num2str(nt),'.mat']));
        fn_gt = setxor(1:1:nSource,tp_gt);
        figure(3);
        plot3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ro',...
              Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'r^',...
              xxtp,yytp,zztp,'bx',...
              xxfp,yyfp,zzfp,'b^')
        axis([0 96 0 96 0 21]); grid on;
        if isempty(xxfp)
            legend('TP-GT','TP-EST','Location','Southoutside','Orientation','horizontal')
        else
            legend('TP-GT','FN-GT','TP-EST','FP-EST','Location','Southoutside','Orientation','horizontal')
        end
        title(sprintf('Result After Postpro (Est int) %d',nt))
%         hold on; imagesc(I0); hold off
%         pause(0.5)

        % View Final Prediction After Post-pro Est float
        fn_gt = setxor(1:1:nSource,tp_gt);
        figure(4);
        plot3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ro',...
              Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'r^',...
              xxtp'+sxtp,yytp'+sytp,zztp'+sztp,'bx',...
              xxfp+sxfp,yyfp+syfp,zzfp+szfp,'b^')
        axis([0 96 0 96 0 21]); grid on;
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

fprintf('Total %d Images in %d point source case\n',100,nSource);
fprintf('Precision=%.2f%%, Recall=%.2f%%, Jaccard=%.2f%%, F1 socre=%.2f%% \n',...
        mean_precision*100 ,mean_recall*100, mean_jaccard*100, mean_f1_score*100);
toc

%% Save Info
if save_pred_info
    fclose(label);
    fclose(eval);
end

%% Save Current 3D Grid
% filename = ['D:\OneDrive - City University of Hong Kong\3dloc\previous pre\0729_examples\',num2str(nSource),'_',num2str(nt),'.png'];
% frame = getframe(gca); 
% img = frame2im(frame); 
% imwrite(img,filename); 

%% Plot Flux Estimation for 1 Example
% histogram compare gt and prediction
% figure;
% w1 = 0.5; w2 = 0.25;
% bar(flux_total(1,:),'FaceColor',[0.2 0.2 0.5])
% hold on 
% bar(flux_total(2,:),w2,'FaceColor',[0 0.7 0.7])
% legend('true','est')
% title([num2str(nt),' var']);
% set(gcf,'position',[100,100,1200,600]);
% ylim([0,180]);
% % saveas(gcf,[histogram_save_path,'\',num2str(nt),'.fig']);
% % saveas(gcf,[histogram_save_path,'\',num2str(nt),'.png']);
% hold off

% plot the relative error in flux for entire dataset
% figure;
% flux_per = abs(flux_total(1,:)-flux_total(2,:))./flux_total(1,:);
% h1 = histogram(flux_per);
% title(['# of pts ', num2str(nSource)]);
% xlabel('relative error')
% ylabel('# of pts')

% rmse for each pts
%     [loc_x_tp,loc_y_tp,loc_z_tp] = ind2sub(size(A),tp_pred);
%     elx_tp = []; ely_tp = []; elz_tp = [];
%     for i = 1:length(loc_x_tp)
%         elx_tp(i) = elx(loc_x_tp(i),loc_y_tp(i),loc_z_tp(i));
%         ely_tp(i) = ely(loc_x_tp(i),loc_y_tp(i),loc_z_tp(i));
%         elz_tp(i) = elz(loc_x_tp(i),loc_y_tp(i),loc_z_tp(i));
%     end
%     loc_pred = [loc_x_tp+elx_tp; loc_y_tp+ely_tp; loc_z_tp+elz_tp]';
%     loc_gt = [Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1];
%     loc_error = loc_pred-loc_gt;
%     rmse = zeros(50*nSource,1);

    % Remove Boundary pts
%     pred_tmp = pred_tmp(abs(pred_tmp(:,2))~=47.75,:);
%     pred_tmp = pred_tmp(abs(pred_tmp(:,3))~=47.75,:);
%     pred_tmp = pred_tmp(pred_tmp(:,4)~=-19.914,:);
%     max_pred = max(pred_tmp(:,5));
%     pred_tmp = pred_tmp(pred_tmp(:,5)>max_pred*0.05,:);
