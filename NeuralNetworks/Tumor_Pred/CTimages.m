
%https://wiki.cancerimagingarchive.net/display/Public/LUNGx+SPIE-AAPM-NCI+Lung+Nodule+Classification+Challenge

%{
                    ------Meta Data-------
10 contrast-enhanced CT scans will be available as a calibration dataset. 
This dataset is representative of the technical properties (scanner type, 
acquisition parameters, file format) of the test dataset. Participants
should not necessarily consider the lung nodules present in the calibration
cases to be representative of the difficulty level expected in the test set.
The calibration set contains five CT scans with malignant nodules and five
CT scans with benign nodules. The organizers have selected a single nodule
per CT scan for analysis. The location of each nodule is specified in an
associated Excel file that includes case name, the coordinates of the 
approximate nodule centroid, and the ?truth? label (malignant or benign).
Participants are encouraged to calibrate their algorithms by downloading 
the calibration dataset through the TCIA.

% I first downloaded the 10 CT scans (5 benign/5 malignant), each of
% which contain about 250-350 sections per scan. This was too much data, 
% as each pixel array was 512x512, and having an average of 300 sections
% for each of the 10 CT scans, I was looking at 3,000 512x512 matrices.
% 3000*512*512 = 786,432,000. To reduce computation,
%}


numfiles = 250; BE001 = cell(1, numfiles); imcenter=169;
lbound = imcenter-26;  ubound = imcenter+25;
    for k = 1:numfiles  
        if k<10      
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE001/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE001{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE001/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE001{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE001/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE001{k} = dicomread(myfilename);end   
        end
    end
%We  have just 50 512x512 matrices, all the others  empty,let's delete them.
idx = find(cellfun('isempty', BE001));BE001(idx) = [];

numfiles = 349; BE002 = cell(1, numfiles);
imcenter=117; lbound = imcenter-26;  ubound = imcenter+25;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE002/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE002{k} = dicomread(myfilename);end 
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE002/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE002{k} = dicomread(myfilename);end 
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE002/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE002{k} = dicomread(myfilename);end 
        end
    end
idx = find(cellfun('isempty', BE002));BE002(idx) = [];


numfiles = 387; BE006 = cell(1, numfiles);
imcenter=241; lbound = imcenter-26;  ubound = imcenter+25;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE006/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE006{k} = dicomread(myfilename);end 
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE006/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE006{k} = dicomread(myfilename);end 
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE006/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE006{k} = dicomread(myfilename);end 
        end
    end
idx = find(cellfun('isempty', BE006));BE006(idx) = [];

numfiles = 339; BE007 = cell(1, numfiles);
imcenter=194; lbound = imcenter-26;  ubound = imcenter+25;
for k = 1:numfiles  
  if k<10  
  myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE007/00000%d.dcm', k);
  myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
  if IN<ubound && IN>lbound;BE007{k} = dicomread(myfilename);end
  elseif k>=10 && k<100
  myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE007/0000%d.dcm', k);
  myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
  if IN<ubound && IN>lbound;BE007{k} = dicomread(myfilename);end  
  elseif k>=100
  myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE007/000%d.dcm', k);
  myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
  if IN<ubound && IN>lbound;BE007{k} = dicomread(myfilename);end
  end
end
idx = find(cellfun('isempty', BE007));BE007(idx) = [];

numfiles = 351; BE010 = cell(1, numfiles);
imcenter=69; lbound = imcenter-26;  ubound = imcenter+25;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE010/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE010{k} = dicomread(myfilename);end         
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE010/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE010{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/BE010/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;BE010{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', BE010));BE010(idx) = [];

numfiles = 283; LC001 = cell(1, numfiles);
imcenter=135; lbound = imcenter-26;  ubound = imcenter+26;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC001/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC001{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC001/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC001{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC001/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC001{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', LC001));LC001(idx) = [];

numfiles = 283; LC002 = cell(1, numfiles);
imcenter=70; lbound = imcenter-26;  ubound = imcenter+26;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC002/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC002{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC002/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC002{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC002/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC002{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', LC002));LC002(idx) = [];

numfiles = 411; LC003 = cell(1, numfiles);
imcenter=70; lbound = imcenter-26;  ubound = imcenter+26;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC003/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC003{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC003/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC003{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC003/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC003{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', LC003));LC003(idx) = [];

numfiles = 374; LC008 = cell(1, numfiles);
imcenter=70; lbound = imcenter-26;  ubound = imcenter+25;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC008/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC008{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC008/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC008{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC008/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC008{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', LC008)); LC008(idx) = [];

 numfiles = 279; LC009 = cell(1, numfiles);
 imcenter=63; lbound = imcenter-26;  ubound = imcenter+26;
    for k = 1:numfiles  
        if k<10  
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC009/00000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC009{k} = dicomread(myfilename);end
        elseif k>=10 && k<100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC009/0000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC009{k} = dicomread(myfilename);end
        elseif k>=100
        myfilename = sprintf('/Users/dawnstear/desktop/trainCT/LC009/000%d.dcm', k);
        myfileinfo = dicominfo(myfilename);IN = myfileinfo.InstanceNumber;
        if IN<ubound && IN>lbound;LC009{k} = dicomread(myfilename);end
        end
    end
idx = find(cellfun('isempty', LC009)); LC009(idx) = [];


% ROI/patch extraction, 64x64 around nodule center (x,y)
x = 296; y=405; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(BE001{1})~=64
for i=1:length(BE001)
    c = BE001{i}; BE001{i}= c(x1:x2,y1:y2);
end
end
x =268;y =184;x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(BE002{1})~=64
for i=1:length(BE002)
    c = BE002{i}; BE002{i}= c(x1:x2,y1:y2);
end
end
x = 266; y=449; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(BE006{1})~=64
for i=1:length(BE006)
    c = BE006{i}; BE006{i}= c(x1:x2,y1:y2);
end
end
x = 206; y=385; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(BE007{1})~=64
for i=1:length(BE007)
    c = BE007{i}; BE007{i}= c(x1:x2,y1:y2);
end
end
x = 336; y=120; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(BE010{1})~=64
for i=1:length(BE010)
    c = BE010{i}; BE010{i}= c(x1:x2,y1:y2);
end
end
%%%%%%% malignant tumors
x = 325; y=120; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(LC001{1})~=64
for i=1:length(LC001)
    c = LC001{i}; LC001{i}= c(x1:x2,y1:y2);
end
end

x = 359; y=139; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(LC002{1})~=64
for i=1:length(LC002)
    c = LC002{i}; LC002{i}= c(x1:x2,y1:y2);
end
end
x = 323; y=375; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(LC003{1})~=64
for i=1:length(LC003)
    c = LC003{i}; LC003{i}= c(x1:x2,y1:y2);
end
end
x = 328; y=95; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(LC008{1})~=64
for i=1:length(LC008)
    c = LC008{i}; LC008{i}= c(x1:x2,y1:y2);
end
end
x = 299; y=145; h=32; x1 = x-h; x2=x+(h-1);y1 = y-h; y2=y+(h-1);
if length(LC009{1})~=64
for i=1:length(LC009)
    c = LC009{i}; LC009{i}= c(x1:x2,y1:y2);
end
end



% Combine all benign into cell array and give them label=1
b = [BE001 BE002 BE006 BE007 BE010];
b(2,:) = {1};
% Combine all malignant into cell array and give them label=0
m = [LC001 LC002 LC003 LC008 LC009];
m(2,:) = {0};

% combine both types
train = [b m];
% column shuffle
train = train(:,randperm(length(train)));
% save as mat files to load into python
save('train.mat','train');


