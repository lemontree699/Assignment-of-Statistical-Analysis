% 方差分析
% 读取数据
file = fopen('data.txt');
M = textscan(file, '%f %s %f %f %s %s %f', 'delimiter', ',', 'HeaderLines', 1);
fclose(file);

% 处理数据
sex = M(2);
smoker = M(5);
charges = cell2mat(M(7)); 
sex_str = strings(1338,1);
smoker_str = strings(1338,1);
for i = 1:1338
    sex_str(i) = char(sex{1}{i});
    smoker_str(i) = char(smoker{1}{i});
end

% % 将男性与女性的医疗费用分别存储
% female_charges = zeros(1,1);
% male_charges = zeros(1,1);
% 
% for i = 1:1338
%     if (sex{1}{i} == "female")
%         female_charges = horzcat(female_charges, charges(i));
%     end
%     if (sex{1}{i} == "male")
%         male_charges = horzcat(male_charges, charges(i));
%     end
% end
% 
% female_charges = female_charges';
% male_charges = male_charges';
% female_charges(1) = [];
% male_charges(1) = [];

% 单因素方差分析
p1 = anova1(charges,sex_str);

% 双因素方差分析
% p2 = anovan(charges,{sex_str, smoker_str});
p2 = anovan(charges,{sex_str smoker_str},'model', 'interaction', 'varnames', {'sex_str', 'smoker_str'});

