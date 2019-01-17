clear;
% im1 = imread('./att_faces/s39/1.pgm');
% imshow(im1);
% im1 = reshape(im1, [1,112*92]);

% 将训练数据存储在矩阵train_data中，有240行，代表240张图片，每行即每张图片表示成一个112*92维向量。
train_data = zeros(240,112*92);
att_faces = dir('./att_faces');
count = 1;
for i=4:43
    file = dir(['./att_faces/', att_faces(i).name]);
    for j=[3,5,6,7,8,9]
        temp = imread(['./att_faces/', att_faces(i).name, '/', file(j).name]);
        train_data(count,:) = reshape(temp, [1,112*92]);
        count = count + 1;
    end
end

% 将测试数据存储在矩阵test_data中，有160行，代表160张图片，每行即每张图片表示成一个112*92维向量。
test_data = zeros(160,112*92);
att_faces = dir('./att_faces');
count = 1;
for i=4:43
    file = dir(['./att_faces/', att_faces(i).name]);
    for j=[10,11,12,4]
        temp = imread(['./att_faces/', att_faces(i).name, '/', file(j).name]);
        test_data(count,:) = reshape(temp, [1,112*92]);
        count = count + 1;
    end
end

% 主成分分析
% coeff为原矩阵的协方差矩阵的特征向量，特征值中选最大的k(=40)个，再选择对应的k列特征向量。
% 投影矩阵为coeff前k列。
% score为进行pca压缩后的数据。
% latent为从大到小排序的特征值。
% explained为每个主成分对应的贡献比例。
% mu为原矩阵每列的均值，取消中心化操作之后mu为0。
% [coeff,score,latent,tsquared,explained,mu] = pca(train_data, 'Centered', false);
[coeff,score,latent,tsquared,explained,mu] = pca(train_data);

% 投影矩阵
projection_matrix = coeff(:,1:40);
% 根据投影矩阵计算pca压缩后的结果
train_data_pca = train_data * projection_matrix;

% 利用训练集计算的投影矩阵和均值，对测试集的每一张图（10304维度）降维到40维度。得到160*40矩阵。160是图像的个数，40是维度。
test_data_pca = test_data * projection_matrix;


% 人脸识别与匹配
% 欧式距离
euclidean_distance = dist(train_data_pca, test_data_pca');
[~,euclidean_index] = min(euclidean_distance);
euclidean_collect_num = 0;
for i=1:160
    if strcmp(att_faces(ceil(euclidean_index(i)/6)+3).name, att_faces(ceil(i/4)+3).name)
        euclidean_collect_num = euclidean_collect_num + 1;
    end
end
euclidean_accuracy = euclidean_collect_num / 160;
disp(['The accuracy obtained by Euclidean distance matching method is ', num2str(euclidean_accuracy*100), '%']);
