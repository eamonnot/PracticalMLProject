# Step 1: Load Libraries
library(caret)
library(doParallel)
registerDoParallel(cores=2)

# Step 2 - Get and load the data
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

readInTrainingData <- function(){
  filename <- "data/pml-training.csv"
  training <- read.table(filename, sep=",", skip=0, header=TRUE, 
                         na.strings=c("NA","NaN",""," ","#DIV/0!"))
  return (training)
}

checkAndDownloadFiles()
allTrainingData <- readInTrainingData() 

# Step 3 - Partition training & test datasets
createTrainTest <- function(theData){
  set.seed(41282)
  inTrain <- createDataPartition(y = theData$classe, p=.70, list=FALSE)
  training <- theData[inTrain,]
  testing <- theData[-inTrain,]
  return (list(training = training, testing = testing))
}

datasets <- createTrainTest(allTrainingData)
training <- datasets$training
# Step 4 - Process the data 
# A. Remove columns that are mostly NA
training <- training[,colSums(is.na(training)) < nrow(training) * 0.5]
any(is.na(training))
# B. Next Remove columsn relating to Index, timestamps and windows
training <- training[,-(1:7)]
# C. Check for highly correlated features and remove unnecessary features
descrCor <- cor(training[,-ncol(training)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
training <- training[,-highlyCorDescr]
  
  
# Step 5 - Training
lastCol <- ncol(training)
preProc.Norm <- preProcess(training[,-lastCol],method=c("center","scale"))
  
train.Norm <- predict(preProc.Norm, training[,-lastCol])
  
set.seed(360)
if(!file.exists("model_Norm.Rds")){
  print("Model file for Center+Scale not found. About to train new model. 
          Go get a coffee, this will take a while")
  modelFit.Norm <- train(training$classe ~., method="rf" ,data=train.Norm,
                      trControl=trainControl(method="cv",number=5),
                      prox=TRUE,allowParallel=TRUE)
  saveRDS(modelFit.Norm, "model_Norm.Rds")
  registerDoParallel(cores=1)
}else{
  print("Loading Ceneter+Scale model")
  modelFit.Norm <- readRDS("model_Norm.Rds") 
}
  
  
print("FinalModel")
print(modelFit.Norm$finalModel)
  

# Step 6 - Apply to test partition
testing <- datasets$testing
testing <- testing[,names(training)]
test.Norm <- predict(preProc.Norm, testing[,-lastCol])
preds <- predict(modelFit.Norm, test.Norm)
testing$predRight <- preds == testing$classe

table(preds, testing$classe)

print(nrow(testing[testing$predRight == TRUE,]) / nrow(testing))