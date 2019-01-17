clear;
% im1 = imread('./att_faces/s39/1.pgm');
% imshow(im1);
% im1 = reshape(im1, [1,112*92]);

% 将所有图片存储在矩阵data中，有400行，代表400张图片，每行即每张图片表示成一个112*92维向量。
data = zeros(400,112*92);
att_faces = dir('./att_faces');
count = 1;
for i=4:43
    file = dir(['./att_faces/', att_faces(i).name]);
    for j=3:12
        temp = imread(['./att_faces/', att_faces(i).name, '/', file(j).name]);
        data(count,:) = reshape(temp, [1,112*92]);
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
% [coeff,score,latent,tsquared,explained,mu] = pca(data, 'Centered', false);
[coeff,score,latent,tsquared,explained,mu] = pca(data);

% 投影矩阵
projection_matrix = coeff(:,1:40);
% 根据投影矩阵计算pca压缩后的结果
data_pca = data * projection_matrix;

%k-means(利用降维后的特征进行聚类)
disp('Clustering by Reduced Dimension Features:');
% 每张图的标签
labels = zeros(1,400);
for i = 1:400
    labels(i) = ceil(i/10);
end
% 选取初始类
cluster = zeros(40,40);  % 每一行代表一个类
for i = 1:40
    cluster(i,:) = data_pca(10*(i-1)+1,:);
end
number_of_iterations = 0;
% 开始迭代
while 1
    % 计算每张图应该归于哪个类
    distance = zeros(40,400);  % distance的每一列代表某张图与40个类之间的欧式距离
    for i = 1:40
        distance(i,:) = dist(cluster(i,:), data_pca');
    end
    cluster_belong_to = zeros(1,400);
    for i = 1:400
        [~,kmeans_index] = min(distance(:,i));
        cluster_belong_to(i) = kmeans_index;
    end
    % 算出新的类均值
    new_cluster = zeros(40,40);
    for i = 1:40
        temp = find(cluster_belong_to == i);
        for j = 1:size(temp, 2)
            new_cluster(i,:) = new_cluster(i,:) + data_pca(temp(j),:);
        end
        new_cluster(i,:) = new_cluster(i,:) / size(temp, 2);
    end
    % 给每个类标上标签
    labels_of_cluster = zeros(1,40);
    for i = 1:40
        G = zeros(1,40);  % 记录每一个类中各个标签出现的次数
        temp = find(cluster_belong_to == i);
        for j = 1:size(temp, 2)
            G(labels(temp(j))) = G(labels(temp(j))) + 1;
        end
        [~,t] = max(G);
        labels_of_cluster(i) = t;
    end
    % 如果新算出的类均值和之前的不相等
    if ~isequal(cluster, new_cluster)
        cluster = new_cluster;
        number_of_iterations = number_of_iterations + 1;
        disp(['Now the number of iterations is ', num2str(number_of_iterations)]);
    % 如果新算出的类均值和之前的相等
    else 
        % 计算准确率
        kmeans_collect_num_pca = 0;
        for i = 1:400
            n = cluster_belong_to(i);
            if labels_of_cluster(n) == labels(i)
                kmeans_collect_num_pca = kmeans_collect_num_pca + 1;
            end
        end
        kmeans_accuracy_pca = kmeans_collect_num_pca / 400;
        disp(['Iterative completed. The accuracy obtained by kmeans is ', num2str(kmeans_accuracy_pca*100), '%']);
        break;
    end
end

%k-means(利用图片所有特征进行聚类)
disp('Clustering Using All Characters of Pictures:');
% 每张图的标签
labels = zeros(1,400);
for i = 1:400
    labels(i) = ceil(i/10);
end
% 选取初始类
cluster = zeros(40,112*92);  % 每一行代表一个类
for i = 1:40
    cluster(i,:) = data(10*(i-1)+1,:);
end
number_of_iterations = 0;
% 开始迭代
while 1
    % 计算每张图应该归于哪个类
    distance = zeros(40,400);  % distance的每一列代表某张图与40个类之间的欧式距离
    for i = 1:40
        distance(i,:) = dist(cluster(i,:), data');
    end
    cluster_belong_to = zeros(1,400);
    for i = 1:400
        [~,kmeans_index] = min(distance(:,i));
        cluster_belong_to(i) = kmeans_index;
    end
    % 算出新的类均值
    new_cluster = zeros(40,112*92);
    for i = 1:40
        temp = find(cluster_belong_to == i);
        for j = 1:size(temp, 2)
            new_cluster(i,:) = new_cluster(i,:) + data(temp(j),:);
        end
        new_cluster(i,:) = new_cluster(i,:) / size(temp, 2);
    end
    % 给每个类标上标签
    labels_of_cluster = zeros(1,40);
    for i = 1:40
        G = zeros(1,40);  % 记录每一个类中各个标签出现的次数
        temp = find(cluster_belong_to == i);
        for j = 1:size(temp, 2)
            G(labels(temp(j))) = G(labels(temp(j))) + 1;
        end
        [~,t] = max(G);
        labels_of_cluster(i) = t;
    end
    % 如果新算出的类均值和之前的不相等
    if ~isequal(cluster, new_cluster)
        cluster = new_cluster;
        number_of_iterations = number_of_iterations + 1;
        disp(['Now the number of iterations is ', num2str(number_of_iterations)]);
    % 如果新算出的类均值和之前的相等
    else 
        % 计算准确率
        kmeans_collect_num = 0;
        for i = 1:400
            n = cluster_belong_to(i);
            if labels_of_cluster(n) == labels(i)
                kmeans_collect_num = kmeans_collect_num + 1;
            end
        end
        kmeans_accuracy = kmeans_collect_num / 400;
        disp(['Iterative completed. The accuracy obtained by kmeans is ', num2str(kmeans_accuracy*100), '%']);
        break;
    end
end
