imds = imageDatastore('D:\cihan\320X240 VERİ JPEG', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.75,'randomized');

layers=[
imageInputLayer([240 320 3])
    
    convolution2dLayer([20,5],15,'Padding','same','Stride',3)
    batchNormalizationLayer
    reluLayer
    
     maxPooling2dLayer(3,'Stride',3)
    
    
    convolution2dLayer([20,20],8,'Padding','same','Stride',3)
    batchNormalizationLayer
    reluLayer
    
   
convolution2dLayer([20,20],8,'Padding','same','Stride',3)
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([20,20],8,'Padding','same','Stride',3)
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([20,20],8,'Padding','same','Stride',3)
    batchNormalizationLayer
    reluLayer
    
 

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

  
options=trainingOptions("sgdm","InitialLearnRate",0.001,...
    'Plots',"training-progress",...
     'MaxEpochs',20, ...
    'ExecutionEnvironment','auto');


[net,graph] = trainNetwork(imdsTrain,layers,options)

[predictedLabels,scores] = classify(net,imdsValidation);

[cm,cmNames] = confusionmat(imdsValidation.Labels,predictedLabels);

imdsValidationLabels = imdsValidation.Labels;

confMat = confusionmat(imdsValidationLabels, predictedLabels);
    

sum(diag(confMat))/sum(confMat(:))
mean(diag(confMat))

cm=confusionchart(imdsValidationLabels,predictedLabels);

