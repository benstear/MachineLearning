%
% Ben Stear 4/13/18
% Breast Cancer Relapse Prediction Using Support Vector Machine
%

gse_B1 =geoseriesread('/Users/dawnstear/downloads/GSE7390_series_matrix.txt');
get(gse_B1.Data);
d = gse_B1.Data;
% gse_B1.Header.Samples;
% gse_B1.Header.Series;

% extract relapse information which will be our targets
cell = gse_B1.Header.Samples.characteristics_ch1;
mat = cell(15,:); %  e.rfs: 1, relapse info is in 15th row
targets = zeros(1,198);

for i =1:198 % number of cols
   m = cell2mat(mat(i)); % extract single cell & turn it into mat
    if m(8)=='1'         % 0 or 1 is in 8th position in the string
        targets(i)=1;
    else
        targets(i)=0;
    end
end

rownames = gse_B1.Data.RowNames;

% call fcns containing the 76 (+) and (-) genes
er_plus = get_er_plus();
er_minus = get_er_minus();

% combine into single vector for ease of searching
genes76 = [er_plus er_minus];

% get the indices of the 76 genes we care about
idx = ismember(rownames,genes76);

% filter out the 76 genes we want from the data matrix d
genes = d(idx,:);

% add targets to the end of the data
genes = [genes; targets];

% Create 10-fold partition
c = cvpartition(targets,'KFold',10);



% randomize the columns so we can easily do k-fold cross-validation
random_genes = genes(:,randperm(size(genes, 2))); 
 
% Each iteration we want to use a 90/10 train/test split,
% so lets split the data into 10 subgroups and cycle through,
% we have 198 samples, we can make 9 subgroups of 20 and one of 18
sg1 = random_genes(:,1:20); sg2 = random_genes(:,21:40); 
sg3 = random_genes(:,41:60); sg4 = random_genes(:,61:80);
sg5 = random_genes(:,81:100); sg6 = random_genes(:,101:120);
sg7 = random_genes(:,121:140); sg8 = random_genes(:,141:160);
sg9 = random_genes(:,161:180); sg10 = random_genes(:,181:198);

% Now we want to cycle through the data, each time using 90% to train
% the SVM model and 10% to test it. 

% Combine subgroups into 1 cell array so we 
% can cycle through them iteratively
subgroups = {sg1 sg2 sg3 sg4 sg5 sg6 sg7 sg8 sg9 sg10};

% Initialize Components for SVM training and testing
sprev = rng(0,'v5uniform'); % for reproducibility
r = randperm(10);
train = {}; 
percentCorrect = zeros(1,10); 
true_pos = zeros(1,10);  true_neg = zeros(1,10); 
false_pos = zeros(1,10); false_neg = zeros(1,10);


% MAIN FOR LOOP
for i=1:length(subgroups)
    %pick out 1 subgroup to be the test data,rest will be train data
    test = subgroups{r(i)}; 
    % test is a dataMatrix object with last row being the binary labels,
    % However, we wont need the labels for the test group, (only to
    % evaluate the models accuracy later), so lets store them some place
    % else and then strip them off test matrix.
    test_labels = test(end,:);
    test(end,:) = [];
    
    trainingcells = subgroups(r(r~=r(i))); % get the other 9 subgroups
    train = [trainingcells{:}]; % concat horizontally
    % Now save labels somewhere and then delete them
    train_labels = train(end,:);
    train(end,:) = [];
    
    % convert train datamats into array 
    X_train = single(train)' ;
    y_train = single(train_labels)';
    % convert test datamats into array 
    X_test = single(test)';
    y_test = single(test_labels)';
    
    % Now train SVM
    SVMModel = fitcsvm(X,y);     % (finally)
    
    % Now test SVM
    [label,score] = predict(SVMModel,X_test); %label is logical vector
    
    % Accuracy
    percentCorrect(i) = sum((y_test == label))/ size(test,2);
    
    % TruePosRate = P/(TP+FN)
    true_pos(i) = sum(label & y_test)/...
        (sum(label & y_test)+sum(~label & y_test));
    
    % FalsePosRate = FP/(FP+TN)
    false_pos(i) = sum(label & ~y_test)/...
        (sum(label & ~y_test)+sum(~label & ~y_test));

end


% Total Metrics
Total_ACC = (sum(percentCorrect)/10)*100

%%

% DOPLOT,  plot ROC
doplot = 1;
if doplot
   figure
   plot(sort(false_pos),sort(true_pos));
   xlabel('False Positive Rate'); ylabel('True Positive Rate')
end

%% 

% Find area under ROC curve
AUC = trapz(sort(false_pos), (-1./sort(true_pos)))

%%

% Get relevant matrices/vectors for sequentialfs
XTrain = random_genes(1:(end-1),1:178);
yTrain = random_genes(end, 1:178);
Xtest = random_genes(1:(end-1),178:end);
ytest = random_genes(end,178:end);
X_all = random_genes(1:end-1,1:end);
y_all = random_genes(end,1:end);

fun = @(XTrain,yTtrain,Xtest,ytest)...
      (sum(~strcmp(ytest,classify(Xtrain,XTrain,yTrain,'quadratic'))));
  
% Compute forward feature (gene) selection
% [fs,history] = sequentialfs(fun,X_all,y_all,c); 

% wasnt working correctly^^^

confusionmat(y_test,label)

%%

% To obtain the best AUC, select 6 genes...

random_genes.rownames([8 11 36 42 70 ])

%%

% get sample data for these 6 genes
sixbest  = random_genes([8 11 36 42 70 end],:); % include end for lbls

% split into 90/10
n = floor(size(sixbest,2)*.9); % training size (178)

sixbest_Xtrain = single(sixbest(1:(end-1),1:n))';
sixbest_ytrain = single(sixbest(end,1:n))';

sixbest_Xtest = single(sixbest(1:(end-1),(n+1):end))';
sixbest_ytest = single(sixbest(end,(n+1):end))';

% Now train SVMModel2
SVMModel2 = fitcsvm(sixbest_Xtrain,sixbest_ytrain);  
    
% Now test SVM
[label2,score2] = predict(SVMModel2,sixbest_Xtest); 

% Find accuracy
Accuracy2 = mean(SVMModel2.predict(sixbest_Xtest) == sixbest_ytest) *100

%%

% Accuracy and AUC are both worse, I think I may have messed something up,
% either selecting the wrong genes or the wrong number of genes







