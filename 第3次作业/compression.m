function [information,image_compression] = compression(filename, r)
    image = imread(filename);
    %[row, col] = size(image);  %获取图片像素的行数与列数

    %将图像分为16*16的小块，对于512*512的图片，一共32*32=1024块
    X = zeros(16,16,1024);  %随机变量X1,X2,...,X1024
    region_size=16;
    numRow=32;
    numCol=32;
    t1 = (0:numRow-1)*region_size + 1; t2 = (1:numRow)*region_size;
    t3 = (0:numCol-1)*region_size + 1; t4 = (1:numCol)*region_size;
    %figure; 
    k = 0; 
    for i = 1 : numRow
        for j = 1 : numCol
            temp = image(t1(i):t2(i), t3(j):t4(j), :);
            k = k + 1;
            X(:,:,k) = temp;
    %         subplot(numRow, numCol, k);
    %         imshow(temp);
        end
    end

    %对每一部分的图像做主成分分析
    Y = zeros(16,16,1024);
    total = 0;
    part = 0;
    for i = 1:1024
        [u,s,v] = svd(double(X(:,:,i)));

        %重构压缩后的图像
        %r = 4;    %压缩率
        %K = round(16 / r);
        K =round(2 * region_size * region_size / ( r * (region_size + region_size + 1)));

        if K > region_size
            K = region_size;
        end

        Y(:,:,i) = zeros(size(X(:,:,i)));
        for j = 1:K
            Y(:,:,i) = Y(:,:,i) + s(j,j) * u(:,j) * v(:,j)';    % 利用前K个特征值重构原图像
        end

        %计算信息量
        total = total + sum(diag(s));
        for w = 1:K
            part = part + s(w,w);
        end
    end

    %将小块合成一张图
    image_compression = zeros(1,512);
    for j = 1:32
        image_compression_row = Y(:,:,(j-1)*32+1);
        for k = 2:32
                image_compression_row = cat(2, image_compression_row, Y(:,:,(j-1)*32+k));
        end
        image_compression = cat(1, image_compression, image_compression_row);
    end
    image_compression(1,:) = [];

    %计算信息量
    % total = sum(diag(s));
    % part = 0;
    % for i = 1:K
    %     part = part + s(i,i);
    % end
    information = part / total;

end