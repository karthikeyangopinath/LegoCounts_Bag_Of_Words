
function [numA,numB]= count_lego(I)

set_current_dir=extractBetween(mfilename('fullpath'),1,max(strfind(mfilename('fullpath'),'\')));
cateogory_file_name=strcat(set_current_dir(1),'LegocategoryClassifier.mat');
if isfile(cateogory_file_name)
     Lego_class=load(string(cateogory_file_name));
else
     BagofWords();
     Lego_class=load(string(cateogory_file_name));
end


set_current_dir=extractBetween(mfilename('fullpath'),1,max(strfind(mfilename('fullpath'),'\')));
cateogory_file_name=strcat(set_current_dir(1),'LegocategoryClassifier.mat');
Lego_class=load(string(cateogory_file_name));
% LEGO Count
%_______________
% For counting the legos, I have used the following methods
%1. Segmentation by Colour
%2. Bag of Visual Words - For Class Identification
%3. Count the lego

%Step 1: Segmentation by Color
% We are using the color space to segment the legos into blue and red
% separatrely
%After that we will be using the morphological operators to get information
% get then classified into individual objects and then we use Image
% labeling to label the segmented images

%The indidivual segmented images will be predicted to belong to blue or red
%legos based on the LegoCategoryClassifier built from the BagOfVIsual
%Words.

%Step 2: Bag of Visual Words
%We are following the below steps to count the blue and red legos
%1. Bag of Visual Words: The Bag of Visual words help you to classify
%object into into different categories. In this scenario, we are using it
%to categorise the BagofWords into 
%           a. 2x2 legos
%           b. 2x4 legos
%           c. Others

% The main reason why we are clasifying it based on shape is because we are
% doing colour segmentation, so it it not necesarry to classify them based
% on colour again in BagOfWords. So we only need the shape information for
% classification.

% Bag of Words Feature Extraction: For the feature extract of BagOf Visual
% Words Im using a custom extractor - HOG extractor to extract he features
% about the Images. You can see this in BagOfWords.m files which calls the
% HOG custom Extracor. The HOG Feature Extractor seems to work well for the
% Extraction since the objects are subject to lot of variation to
% illumination

%We are using a custom template SVM where we give the parameters to tune the
%Model and obtain results from that. It is also part of BagOfWords.m file.

%Then we are splitting the images into Training Set and Test Set and then
%testing the images to evaluate the score. This can also be seen when the
%BagOfWords.m file is executed separately.

%After evaluating we are save the CategoryClassifer as
%LegoCategoryClassifer.m file to be used in this function.


%Step 3: Count the legos
% The legos will be counted based on the classifier and will sent as
% output.



im2d=im2double(im2gray(I));


%Implementting Coloured Segmentation
%First Segment Blue colour LEGOS
im2b=im2double(I(:,:,3));
imbin=imbinarize(im2b-im2d);





%Getting the correct shape of the blue legos
se = strel('disk',15);
img_close=imclose(imbin,se);
[img_label,numberOfRegions]=bwlabel(img_close);
%Getting the position of all the blue legos in the BoundingBox parameter
lego_descrip=regionprops(img_label,'Area','BoundingBox');


count_blue_legos=0;
blue_lego_class={};
for i=1 : numberOfRegions
if lego_descrip(i).Area >= 3000
img_cropped=imcrop(I,lego_descrip(i).BoundingBox);
[blue_lego_labels, score] = predict(Lego_class.LegocategoryClassifier,img_cropped);
blue_lego_class(i)=Lego_class.LegocategoryClassifier.Labels(blue_lego_labels);
if strcmp(blue_lego_class(i),'2x4')
     count_blue_legos = count_blue_legos +1;
end
end
end




%Implementing coloured Segmentation in Red Space
%Calling another mask function to get the red blocks identified
im2r=createMask(I);
imbinr=imbinarize(im2r-im2d);

%Getting the correct shape of the lego
se = strel('disk',15);
img_closer=imclose(imbinr,se);
[img_labelr,numberOfRegions_r]=bwlabel(img_closer);
%Getting the position of all the blue legos in the BoundingBox parameter
lego_descrip_r=regionprops(img_labelr,'Area','BoundingBox');



%Here we are getting the classification of whether the lego belong to 2x2,
%2x4 or Others for RED LEGOS

count_red_legos=0;
red_lego_class={};
for i=1 : numberOfRegions_r
if lego_descrip_r(i).Area >= 2000
img_cropped_r=imcrop(I,lego_descrip_r(i).BoundingBox);
[red_lego_labels, score] = predict(Lego_class.LegocategoryClassifier,img_cropped_r);
red_lego_class(i)=Lego_class.LegocategoryClassifier.Labels(red_lego_labels);
if strcmp(red_lego_class(i),'2x2')
     count_red_legos = count_red_legos +1;
end
end
end
count_blue_legos
count_red_legos
numB=count_red_legos;
numA=count_blue_legos;
    
