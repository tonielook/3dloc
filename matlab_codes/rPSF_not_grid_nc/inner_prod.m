function add_mat=inner_prod(h, A)
%     Nz = size(A,3);
    if min(size(h)) == 1 
        h = reshape(h,size(A,1),size(A,2));
    end
%     h_tem = rot90(rot90(h));
%     add_mat= zeros(size(A));
%     for zi =  1:Nz
%         Int_p = ifft2(fft2(A(:,:,zi)).*fft2(h_tem));
%         add_mat(:,:,end-zi+1) = rot90(rot90(Int_p));
%     end
    add_mat= zeros(size(A));
    add_mat(:,:,1) = h(end:-1:1,end:-1:1);
    add_mat = ifftn(fftn(A).*fftn(add_mat));
    add_mat = add_mat(end:-1:1, end:-1:1, end:-1:1);
end
