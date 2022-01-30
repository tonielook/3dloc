%% Post-processing for initial prediction from NN
%% parameters from generating data
load('data_natural_order_A'); % Single role
global Np nSource L Nzones
L = 4; Nzones = 7; b = 5; [Nx,Ny,Nz] = size(A); Np = Nx;
zmax = 20;						

tic
%% Modify parameters here
save_pred_info = 0; % save pred_label.txt
% nSource = 5;

% mat_path = [' ',num2str(nSource)]; % path for test data
% pred_path = ' '; % path for prediction
mat_path = ['../../data_test/test',num2str(nSource)]; % path for test data
pred_path = ['../../test_output/test',num2str(nSource)];
save_path = pred_path;

%% main
pred = readtable([pred_path,'/loc.csv']);
pred = table2array(pred);
gt = readtable([pred_path,'/label.txt']);
gt = table2array(gt(:,1:5));
testsize=size(dir([mat_path '/im*.mat']),1);

% evaluation metrics
recall = zeros(testsize,1);
precision = zeros(testsize,1);
jaccard_index = zeros(testsize,1);
f1_score = zeros(testsize,1);
initial_pred_pts = zeros(testsize,1);
flux_all = [];

if save_pred_info
    % pred_label: pred 3d locations + flux estimated from var/dnn
    label = fopen([save_path,'/pred_label.txt'],'w');
end

for nt = 1:testsize
    %% Post-processing
    gt_tmp = gt(gt(:,1)==nt,:);
    pred_tmp = pred(pred(:,1)==nt,:);

    % remove boundary pts
%     pred_tmp_after = pred_tmp(abs(pred_tmp(:,2))~=47.75,:);
%     pred_tmp_after = pred_tmp_after(abs(pred_tmp_after(:,3))~=47.75,:);
%     pred_tmp = pred_tmp(pred_tmp(:,4)~=-19.914,:);

    % load ground truth 3d grid
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
    
    % load initial prediction
    Vpred = [pred_tmp(:,2);pred_tmp(:,3);pred_tmp(:,4);pred_tmp(:,5)];
    pred_vol = zeros(size(A));
    nPred = length(Vpred)/4;
    for i = 1 : nPred
        xlow = min(round(49+Vpred(i)),96); 
        ylow = min(round(49+Vpred(i+ nPred)),96);
        zlow = min(round((Vpred(i+2*nPred)+21)/2.1)+1,20);
        pred_vol(xlow,ylow,zlow)= pred_vol(xlow,ylow,zlow)+Vpred(i+3*nPred);
    end

    initial_pred_pts(nt) = numel(find(pred_vol>0));
    % Removing the clustered false positive 
    [xIt, elx, ely, elz] = local_3Dmax_large(pred_vol);
    
    % Iterative Scheme on refinment on estimation of flux & Evaluation
    idx_est = find(xIt>0); 
    if isempty(idx_est)
        continue
    end
    
    flux_est_dnn = xIt(idx_est);
    % Refinment on Estimation of Flux
						 
     load(fullfile(mat_path,['im',num2str(nt),'.mat']));  % mat file for g
%     flux_est_var = Iter_flux(A, idx_est, g, b);

    %% Evaluation
    num_gt = nSource; num_pred = length(idx_est);
%     [num_tr,tp_pred,tp_gt,flux_total] = evaluation(xIt, interest_reg, flux_est_dnn, flux_gt);
%     [num_tr,tp_pred,tp_gt,flux_total] = evaluation_v2(xIt, interest_reg, flux_est_dnn, flux_gt,nSource);
     [num_tr,tp_pred,tp_gt,flux_total] = evaluation_v3(gt_tmp(:,2:5),xIt,elx,ely,elz,flux_est_dnn);
																								

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

    % save hard samples if recall < 0.95
    
    if ~exist('../../data_train/hardsamples/train/', 'dir')
        mkdir('../../data_train/hardsamples/train/')
    end

    % if re < 0.95
    % hs_recall_bar=0.95
    if re < hs_recall_bar
       datestring = datestr(now,'mmddHH');
       ns_padded = sprintf('%02d',nSource);
       nt_padded = sprintf('%03d',nt);
       hsfileidx = append('20000',datestring,ns_padded,nt_padded);
       hslabel = gt_tmp;
       hslabel(hslabel==nt) = str2num(hsfileidx);
       dlmwrite('../../data_train/hardsamples/train/label.txt',hslabel,'precision',16,'delimiter',' ','-append');
       save(['../../data_train/hardsamples/train/','im',hsfileidx,'.mat'],'g');
       load ([mat_path,'/I',num2str(nt),'.mat']);
       save(['../../data_train/hardsamples/train/','I',hsfileidx,'.mat'],'I0');
       dlmwrite('../../data_train/hardsamples/summary.csv',{str2num(datestring),testsize,nSource,str2num(hsfileidx),re},'precision',16,'delimiter',',','-append');
    end

   
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
mean(initial_pred_pts)

%% save info
if save_pred_info
    fclose(label);
										
																   
    fclose(eval);
end


%% plot the 3D estimation (compare ground true with estimated solution)
%[loc_x_tp,loc_y_tp,loc_z_tp] = ind2sub(size(A),intersect(tp_pred,find(xIt>0))); 
%[loc_x_fp,loc_y_fp,loc_z_fp] = ind2sub(size(A),setxor(tp_pred,find(xIt>0))); 

% figure;
% % true positive - gt
% scatter3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ko')
% hold on
% % false negative - gt
% fn_gt = setxor(1:1:nSource,tp_gt);
% scatter3(Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'ro')
% hold on
% % true positive - pred
% scatter3(loc_x_tp,loc_y_tp,loc_z_tp,'k^')
% hold on
% % false positive - pred
% scatter3(loc_x_fp,loc_y_fp,loc_z_fp,'r^')
% 
% axis([1 96 1 96 0 21])
% legend('tp-gt','fn-gt','tp-p','fp-p','Location','Southoutside','Orientation','horizontal')
% title(['img',num2str(nt)])
% load ([mat_path,'/I50.mat'])
% imagesc(imrotate(flip(I0,2),90))

% append results
dlmwrite('../../test_output/postpro_result.csv',{nSource,mean_precision,mean_recall,testsize},'delimiter',',','-append');

%% save current 3d grid
% filename = [' '_',num2str(nt),'.png']; % save path
% frame = getframe(gca); 
% img = frame2im(frame); 
% imwrite(img,filename); 

%% plot flux estimation for 1 example
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

