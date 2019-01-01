clear;
clc;

image = imread('lena.bmp');

[information1,image_compression1] = compression('lena.bmp', 2);
[information2,image_compression2] = compression('lena.bmp', 4);
[information3,image_compression3] = compression('lena.bmp', 8);

%显示图像
figure;
subplot(2,2,1);
imshow(image,[]);
title('原始图像');
subplot(2,2,2);
imshow(uint8(image_compression1),[]);
title(['压缩为', num2str(1/2), ' 信息量:', num2str(information1)]);
subplot(2,2,3);
imshow(uint8(image_compression2),[]);
title(['压缩为', num2str(1/4), ' 信息量:', num2str(information2)]);
subplot(2,2,4);
imshow(uint8(image_compression3),[]);
title(['压缩为', num2str(1/8), ' 信息量:', num2str(information3)]);
%imwrite(uint8(image_compression), 'lena_compression.bmp');