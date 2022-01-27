function [num_tr,tp_pred,tp_gt,flux_total] = evaluation_v3(gt,xIt,elx,ely,elz,flux_est)
% xIt: 3d grid with nonzero entries are predicted pts
% elx, ely, elz: 3d grid with shift in x, y and z axis
% gt: x,y,z coordinates and flux information for each gt pts, nSource-by-5
% flux_est: estimated flux values from dnn/var
idx_est = find(xIt>0);
[xx,yy,zz] = ind2sub(size(xIt),idx_est);
sx=zeros(length(xx),1);  sy=zeros(length(xx),1); sz=zeros(length(xx),1);
for sidx = 1: numel(idx_est)
    tx = xx(sidx); ty = yy(sidx); tz = zz(sidx);
    sx(sidx) = elx(tx, ty, tz);
    sy(sidx) = ely(tx, ty, tz);
    sz(sidx) = elz(tx, ty, tz);
end

est = [idx_est,xx+sx-49,yy+sy-49,(zz+sz-1)*2.1-21,flux_est];
% est = [idx_est,xx+sx-49,yy+sy-49,zz+sz-21,flux_est];
gt = [(1:size(gt,1))',gt];

num_nonz = size(gt,1);
num_tr = 0;
flux_total = []; 
tp_pred = []; tp_gt = [];

i=1;
while num_nonz>0
    dist = abs(est(:,2:4)-gt(i,2:4));
    tem = ((dist(:,1)<=2)+(dist(:,2)<=2)+(dist(:,3)<=2))==3;
    if sum(tem)>1 && i<num_nonz
        i=i+1;
        continue
    elseif sum(tem)~=0
        num_tr = num_tr + 1;
        idx_con = est(tem,1); idx_con = idx_con(1);
        tp_pred = [tp_pred, idx_con]; tp_gt = [tp_gt,gt(i,1)];
        flux_tp = est(tem,5); flux_add = [gt(i,5); flux_tp(1)];
        flux_total = [flux_total flux_add];
        est(est(:,1)==idx_con,:) = [];
        gt(i,:)=[];
    elseif sum(tem)==0
        gt(i,:)=[];
    end
    num_nonz = num_nonz-1;
    if num_nonz<i
        i = 1;
    end
end

% Add the fasle postive in the estimated result flux_est 
for i = 1:size(est,1)
    flux_add = [0; est(i,5)];
    flux_total = [flux_total flux_add];
end

end

