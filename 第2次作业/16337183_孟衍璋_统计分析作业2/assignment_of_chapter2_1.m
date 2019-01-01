% 回归分析
% 读取数据
file = fopen('data.txt');
M = textscan(file, '%f %s %f %f %s %s %f', 'delimiter', ',', 'HeaderLines', 1);
fclose(file);

% 处理数据
age_full = cell2mat(M(1));
age = age_full(1:1333);
bmi_full = cell2mat(M(3));
bmi = bmi_full(1:1333);
children_full = cell2mat(M(4));
children = children_full(1:1333);
charges_full = cell2mat(M(7)); 
charges = charges_full(1:1333);

% 线性回归
X = [ones(length(charges),1), age, bmi, children];
[b,bint,r,rint,stats] = regress(charges,X,0.05);

% 验证
age_check = age_full(1334:1338);
bmi_check = bmi_full(1334:1338);
children_check = children_full(1334:1338);
charges_check = charges_full(1334:1338);

charges_estimate = b(1) + b(2) * age_check + b(3) * bmi_check + b(4) * children_check;

% 置信度为95%的置信区间
confidence_interval_down = bint(1,2) + bint(2,1) * age_check + bint(3,1) * bmi_check + bint(4,1) * children_check;
confidence_interval_up = bint(1,1) + bint(2,2) * age_check + bint(3,2) * bmi_check + bint(4,2) * children_check;
 