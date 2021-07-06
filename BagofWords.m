function  BagofWords()
%Bag of Visual Words
% Setting up the path variable
%-------------------------------
%  Determine where your m-file's folder is.
folder = fileparts(which(mfilename)); 
% Add that folder plus all subfolders to the path.
addpath(genpath(folder));
imds=imageDatastore(strcat(extractBetween(mfilename('fullpath'),1,max(strfind(mfilename('fullpath'),'\'))),'/data_set/'),'IncludeSubfolders',true,'LabelSource','foldernames');


%Custom Extractor using HOG
extractorFcn = @BagOfFeaturesExtractor;

%Splitting it into Training and test sets
[trainingSet,testSet] = splitEachLabel(imds,0.6,'randomize');

%Generating the Bag of Visual Words
bag = bagOfFeatures(trainingSet,'CustomExtractor',extractorFcn); %'GridStep',[3,3],'VocabularySize',5,'StrongestFeatures',1
imageIndex = indexImages(imds,bag);
featureVector=encode(bag,imds);


%Fine Tuned parameter settings for the SVM to be used for Bag of Words
opts =templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 79.02896058326633, ...
    'BoxConstraint', 1.320844967583944, ...
    'Standardize', true);

%Train the classifier using this Bag of words
LegocategoryClassifier = trainImageCategoryClassifier(imds,bag,'LearnerOptions',opts);


%categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confMatrix = evaluate(LegocategoryClassifier,trainingSet);
mean(diag(confMatrix))

% Set the Working Directory to save the classifier
set_current_dir=extractBetween(mfilename('fullpath'),1,max(strfind(mfilename('fullpath'),'\')));
cateogory_file_name=strcat(set_current_dir(1),'LegocategoryClassifier.mat');
%Save the Classifier
save(string(cateogory_file_name));
