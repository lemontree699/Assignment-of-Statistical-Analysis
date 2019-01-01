function oword=OCRdb(imagen)


if size(imagen,3)==3 %RGB image
    imagen=rgb2gray(imagen);
end
threshold = graythresh(imagen);
imagen =~im2bw(imagen,threshold);
imagen = bwareaopen(imagen,30);
word=[ ];
re=imagen;
fid = fopen('text.txt', 'wt');
load templates
global templates
num_letras=size(templates,2);
while 1
    [fl re]=lines(re);
    imgn=fl;
    [L Ne] = bwlabel(imgn);    
    for n=1:Ne
        [r,c] = find(L==n);
        n1=imgn(min(r):max(r),min(c):max(c));  
        img_r=imresize(n1,[42 24]);
        letter=read_letter(img_r,num_letras);
        word=[word letter];
    end
    fprintf(fid,'%s\n',word);
    oword=word;
    word=[ ];
    if isempty(re) 
        break
    end    
end
fclose(fid);
%winopen('text.txt')
oword;
end
