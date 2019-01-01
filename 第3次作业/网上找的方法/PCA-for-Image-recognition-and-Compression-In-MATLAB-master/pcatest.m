% Im_pca is any square input image matrix.
I=imread('lena.bmp');
Im_pca=double(I);
[M,N] = size(Im_pca);
mn = mean(Im_pca,1);
%means=repmat(mn,M,1);
Im_pca_z=Im_pca-repmat(mn,M,1);
disp('zero mean matrix done!')
 
 for y=1:1:N
    for in=1:1:N
            covar_temp(y,in)=1/((M*N)-1)*(Im_pca_z(:,y)' * Im_pca_z(:,in))  ;
    end
end
 
[PC, V] = eig(covar_temp);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);


ret=input('no.of. principal components to retain');
output_data = PC(:,1:(ret))' * Im_pca_z';
disp('PCA done!')
 
% Reconstruction
reconstruc_data=round((PC(:,1:(ret))*output_data)+repmat(mn,M,1)');
imwrite(reconstruc_data, 'a.bmp');
disp('Reconstruction done!');

