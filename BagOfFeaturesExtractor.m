function [features,featureMetrics,location] = BagOfFeaturesExtractor(img)

im2d=im2double(im2gray(img));

detector=detectMinEigenFeatures(im2d);
pos1=detector.Location;
[features,validPoints] = extractHOGFeatures(im2d,pos1,'BlockSize',[4,4]);


featureMetrics=[];
for i=1 : size(features+1,1)
    featureMetrics(i,:)=1;
end

location=validPoints;

    