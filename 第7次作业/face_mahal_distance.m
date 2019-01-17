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
% coeff为原矩阵的协方差矩阵的特征向量，特征值中选最大的k(=4)个，再选择对应的k列特征向量。
% 投影矩阵为coeff前k列。
% score为进行pca压缩后的数据。
% latent为从大到小排序的特征值。
% explained为每个主成分对应的贡献比例。
% mu为原矩阵每列的均值，取消中心化操作之后mu为0。
% [coeff,score,latent,tsquared,explained,mu] = pca(train_data, 'Centered', false);
[coeff,score,latent,tsquared,explained,mu] = pca(train_data);

% 投影矩阵
projection_matrix = coeff(:,1:4);
% 根据投影矩阵计算pca压缩后的结果
train_data_pca = train_data * projection_matrix;

% 利用训练集计算的投影矩阵和均值，对测试集的每一张图（10304维度）降维到4维度。得到160*4矩阵。160是图像的个数，4是维度。
test_data_pca = test_data * projection_matrix;


% 人脸识别与匹配
% 马氏距离
% 把每个ID分为一类
train_data_mahal = zeros(6,4,40);
for i=1:40
    train_data_mahal(:,:,i) = train_data_pca((i-1)*6+1:i*6,:);
%     train_data_mahal(:,:,i) = train_data_mahal(:,:,i)';
end
% 计算每个ID的均值
mu_mahal = zeros(40,4);
for i=1:40
    mu_mahal(i,:) = mean(train_data_mahal(:,:,i));
end
% 计算马氏距离并分类
mahal_distance = zeros(160,40);
for i = 1:40
    mahal_distance(:,i) = mahal(test_data_pca, train_data_mahal(:,:,i));
end
class_mahal = zeros(1,160);
for i=1:160
    [~,mahal_index] = min(mahal_distance(i,:));
    class_mahal(i) = mahal_index;
end
% 计算准确率
mahal_collect_num = 0;
for i=1:160
    if strcmp(att_faces(class_mahal(i)+3).name, att_faces(ceil(i/4)+3).name)
        mahal_collect_num = mahal_collect_num + 1;
    end
end
mahal_accuracy = mahal_collect_num / 160;
disp(['The accuracy obtained by mahal distance matching method is ', num2str(mahal_accuracy*100), '%']);

% % 马氏距离
% % 把每个ID分为一类
% train_data_mahal = zeros(6,40,40);
% for i=1:40
%     train_data_mahal(:,:,i) = train_data_pca((i-1)*6+1:i*6,:);
% %     train_data_mahal(:,:,i) = train_data_mahal(:,:,i)';
% end
% % 计算每个ID的均值
% mu_mahal = zeros(40,40);
% for i=1:40
%     mu_mahal(i,:) = mean(train_data_mahal(:,:,i));
% end
% % 计算每个ID的协方差以及协方差的逆
% cov_mahal = zeros(40,40,40);
% cov_inv_mahal = zeros(40,40,40);
% for i=1:40
% %     cov_mahal(:,:,i) = (1/5) * cov(train_data_mahal(:,:,i));
%     cov_mahal(:,:,i) = (1/5) * (train_data_mahal(:,:,i) - mu_mahal(i,:))' * (train_data_mahal(:,:,i) - mu_mahal(i,:));
% %     cov_inv_mahal(:,:,i) = pinv(cov_mahal(:,:,i));
% end
% I = eye(40);
% for i=1:40
%     cov_mahal(:,:,i) = (cov_mahal(:,:,i)+I)*0.001;
% end
% for i=1:40
%     cov_inv_mahal(:,:,i) = pinv(cov_mahal(:,:,i));
% end
% % 判别函数
% V = zeros(40,40,160);
% for n=1:160
%     for i=1:40
%         for j = 1:40
%             V(i,j,n) = ((test_data_pca(n,:) - mu_mahal(i,:)) * cov_inv_mahal(:,:,i) * (test_data_pca(n,:) - mu_mahal(i,:))') - ((test_data_pca(n,:) - mu_mahal(j,:)) * cov_inv_mahal(:,:,j) * (test_data_pca(n,:) - mu_mahal(j,:))');
%         end
%     end
% end
% % 根据判别函数决定类别
% class_mahal = zeros(1,160);
% for n = 1:160
%     for j = 1:40
%         if all((V(:,j,n) > 0) == 0)
%             class_mahal(n) = j;
%             break;
%         end
%     end
% end
% % 计算准确率
% mahal_collect_num = 0;
% for i=1:160
%     if strcmp(att_faces(class_mahal(i)+3).name, att_faces(ceil(i/4)+3).name)
%         mahal_collect_num = mahal_collect_num + 1;
%     end
% end
% mahal_accuracy = mahal_collect_num / 160;







