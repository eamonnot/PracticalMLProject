library(caret)

checkAndDownloadFiles <- function(){
  ## First check if data folder exists
  if (!dir.exists("data")){
    dir.create(path = "data")
  }
  if (!file.exists("data/pml-training.csv")){
    fileURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(url = fileURL, destfile = "data/pml-training.csv", method="curl")
  }
  if (!file.exists("data/pml-testing.csv")){
    fileURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url = fileURL, destfile = "data/pml-testing.csv", method="curl")
  }
}

readInData <- function(){
  filename <- "data/pml-training.csv"
  training <- read.table(filename, sep=",", skip=0, header=TRUE, 
                         na.strings=c("NA","NaN",""," ","#DIV/0!"))
  return (training)
}

createTrainTest <- function(theData){
  set.seed(41282)
  inTrain <- createDataPartition(y = theData$classe, p=.70, list=FALSE)
  training <- theData[inTrain,]
  testing <- theData[-inTrain,]
  return (list(training = training, testing = testing))
}

  
  # Step 1 - Check for and download raw data if necessary
  checkAndDownloadFiles()
  
  # Step 2 - Read in the training data
  allData <- readInData() 
  
  # Step 3 - create the training & test datasets
  datasets <- createTrainTest(allData)
  
  # Step 4 - Process the data by removing poor features
  training <- datasets$training
    # A. Remove columns that are mostly NA
  training <- training[,colSums(is.na(training)) < nrow(training) * 0.5]
    # B. Next Remove columsn relating to Index, timestamps and windows
  training <- training[,-(1:7)]
    # C. Check for highly correlated features and remove unnecessary features
  descrCor <- cor(training[,-ncol(training)])
  highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
  training <- training[,-highlyCorDescr]
  
  
  # Step 5 - Train the model
  lastCol <- ncol(training)
  preProc.Norm <- preProcess(training[,-lastCol],method=c("center","scale"))
  
  # Fit the models
  train.Norm <- predict(preProc.Norm, training[,-lastCol])
  
  set.seed(360)
  if(!file.exists("model_Norm.Rds")){
    print("Model file for Center+Scale not found. About to train new model. 
          Go get a coffee, this will take a while")
    library(doParallel)
    registerDoParallel(cores=2)
    modelFit.Norm <- train(training$classe ~., method="rf" ,data=train.Norm,
                           trControl=trainControl(method="cv",number=5),
                           prox=TRUE,allowParallel=TRUE)
    saveRDS(modelFit.Norm, "model_Norm.Rds")
    registerDoParallel(cores=1)
  }else{
    print("Loading Ceneter+Scale model")
    modelFit.Norm <- readRDS("model_Norm.Rds") 
  }
  
  print("The Model")
  modelFit.Norm
  print("FinalModel")
  modelFit.Norm$finalModel
  
  cmTrain <- confusionMatrix(training$classe,predict(modelFit.Norm,train.Norm))
  print(cmTrain$overall)
  print(cmTrain$table)
  
  # Step 6 - Evaluate on the test set
  testing <- datasets$testing
  testing <- testing[,names(training)]
  test.Norm <- predict(preProc.Norm, testing[,-lastCol])
  preds <- predict(modelFit.Norm, test.Norm)
  testing$predRight <- preds == testing$classe
  table(preds, testing$classe)
  
  cmTest <- confusionMatrix(testing$classe,predict(modelFit.Norm,test.Norm))
  print(cmTest$overall)
  print(cmTest$table)
  print(nrow(testing[testing$predRight == TRUE,]) / nrow(testing))
  

