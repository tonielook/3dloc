function sol=  IRL1_l2(g,A,b,a, mu,nu,lambda)
% IRL1 for L2-NC


Nout_iter = 2;
Nin_iter = 400;

y = g(:)';
[ny, nx, nz] = size(A);
T = zeros(1, nz); T(nz) = 1;
fx = zeros(size(A));
feta0 = fx; feta1 = fx;
TT = (T'*T+mu*eye(nz));
fA = fftn(A);
invall = abs(fA).^2+nu/mu;
u0_2d = T'*(-ones(1,ny*nx)*(1+mu*b)+sqrt(ones(1,ny*nx)*(1-mu*b)^2+4*mu*y))/(2*mu);
u0 = reshape(u0_2d', ny, nx, nz);
fu0 = fftn(u0);
u1 = zeros(size(A));
fu1 = fftn(u1);

for oi = 1 : Nout_iter
    
    absx = abs(u1);
    weights = a*lambda/((a+absx).^2);
    for ii = 1:Nin_iter
        uold = u1;
        ftemp_1 = conj(fA).*(fu0-feta0);
        fx = (ftemp_1+nu/mu*(fu1-feta1))./invall;% update x, equation (7)
        fAx = fA.*fx;
        feta0 = feta0-(fu0-fAx);% update eta0, equation (8)
        feta1 = feta1-(fu1-fx);% update eta1, equation (9)
        Ax_eta_2d = ifftn(fAx+feta0);
        if nz > 1 
            Ax_eta_2d = reshape(XYZ_rot(Ax_eta_2d, [3,1,2]), nz, ny*nx);
        else
            Ax_eta_2d = Ax_eta_2d(:)';
        end
        TT1 = T'*ones(1,ny*nx); TT1 =  mu* Ax_eta_2d - (1+mu*b)*TT1;
        u0_2d = TT\(T'*y+mu*Ax_eta_2d);
        u0 = reshape(u0_2d', ny, nx, nz);
        fu0 = fftn(u0);
        x_eta1 = ifftn(fx+feta1);
        u1 = max(x_eta1-weights./nu, 0);% update u1, equation (6) % for ADMM_deconv.m
        fu1 = fftn(u1);
        
        rel_error = norm(u1(:)-uold(:))/(norm(uold(:))+eps);
%         if ii/100 == round(ii/100)
%         fprintf('Rel_error: %2.1e, Outer: %2d, Inner: %2d; \n',...
%             rel_error,oi, ii);
%         end

    end          
        if rel_error < 1e-5
            break;
        end           

end

u1 = ifftshift(u1);
u1 = fftshift(u1,3);
sol = u1;