% Import features
features = readtable("/Users/ilanadeutsch/Desktop/features.csv");
erodedFeatures = readtable("/Users/ilanadeutsch/Desktop/featuresEroded.csv");

iccArray = [];

% Iterature through each feature
for i = 2:size(features,2)

    % Create m matrix
    tmpFeatName = features.Properties.VariableNames(i);
    tmpFeatVals = features.(tmpFeatName{1});
    tmpErodeVals = erodedFeatures.(tmpFeatName{1});
    m = [tmpFeatVals tmpErodeVals];

    % Calculate ICC values
    iccArray = ICC(m, 'A-1', 0.05);
    iccArray = [iccArray; convertCharsToStrings(tmpFeatName) iccVal];

end

% Find icc values above a certain cutoff
iccArray(find(str2double(iccArray(:,2)) > 0.75))

% Export values
writematrix(iccArray,"/Users/ilanadeutsch/Desktop/ICC.csv")







