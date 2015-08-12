mainProcess <- function() {
  library(caret)
  checkAndDownloadFiles()
  allData <- readInData() 
  datasets <- createTrainTest(allData)
  
  trainingSet <- datasets$training
  t <- trainingSet[,colSums(is.na(trainingSet)) < nrow(trainingSet) * 0.5]
  
  t <- t[,-2]
  t <- t[,-2]
  t <- t[,-2]
  t <- t[,-2]
  t <- t[,-2]
  t <- t[,-2]
  
  descrCor <- cor(t[,-54])
  highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
  t.filtered <- t[,-highlyCorDescr]
  
  
  library(doParallel)
  registerDoParallel(cores=2)
 
  t <- t.filtered
  lastCol <- ncol(t)
  
  preProc <- preProcess(t[,-lastCol],method=c("center","scale","pca"), thresh = 0.9)
  #trainPC <- predict(preProc, t[,-lastCol])
  #modelFitLDA <- train(t$classe ~., method="lda" ,data=trainPC)
  
  #cmLda <- confusionMatrix(t$classe,predict(modelFitLDA,trainPC))
  #print("LDA")
  #print(cmLda$overall)
  
  #saveRDS(modelFitLDA, "my_model_lda.Rds")
  
  #modelFitLDA <- train(t$classe ~., method="rpart" ,data=trainPC)
  
  #cmLda <- confusionMatrix(t$classe,predict(modelFitLDA,trainPC))
  #print("RPart")
  #print(cmLda$overall)
  
  #saveRDS(modelFitLDA, "my_model_RPart.Rds")
  
  #modelFitLDA <- train(t$classe ~., method="rf" ,data=trainPC)
  
  #cmLda <- confusionMatrix(t$classe,predict(modelFitLDA,trainPC))
  #print("RF")
  #print(cmLda$overall)
  
  #saveRDS(modelFitLDA, "my_model_RF.Rds")
  
  #modelFitLDA <- train(t$classe ~., method="gbm" ,data=trainPC, verbose=FALSE)
  
  #cmLda <- confusionMatrix(t$classe,predict(modelFitLDA,trainPC))
  #print("GBM")
  #print(cmLda$overall)
  
  #saveRDS(modelFitLDA, "my_model_GBM.Rds")
  
  ##saveRDS(modelFitLDA, "my_model_lda.Rds")
  ##process
  
  testingSet <- datasets$testing
  testing <- testingSet[,colSums(is.na(testingSet)) < nrow(testingSet) * 0.5]
  
  testing <- testing[,-2]
  testing <- testing[,-2]
  testing <- testing[,-2]
  testing <- testing[,-2]
  testing <- testing[,-2]
  testing <- testing[,-2]
  
  testing <- testing[,-highlyCorDescr]
  #testing.filtered2 <- testing.filtered[,-colsToRemove]
  
  testPc <- predict(preProc,testing[,-lastCol])
  print("Here Now")
  my_model_file <- "my_model_GBM.Rds"
  my_model <- readRDS(my_model_file)
  
  cmLda <- confusionMatrix(testing$classe,predict(my_model,testPc))
  print("GBM")
  print(cmLda$overall)
  
  my_model_file <- "my_model_RF.Rds"
  my_model <- readRDS(my_model_file)
  print(varImp(my_model))
  cmLda <- confusionMatrix(testing$classe,predict(my_model,testPc))
  print("RF")
  print(cmLda$overall)
  
  return(t)
  
  #modelFitBoost <- train(t$classe ~., method="gbm" ,data=trainPC, verbose=FALSE)
  
  #cmBoost <- confusionMatrix(testing$classe,predict(modelFitBoost,testPc))
  #saveRDS(modelFitBoost, "my_model_boost.Rds")
  #print(cmBoost$overall)
  
}

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
  set.seed(3223)
  inTrain <- createDataPartition(y = theData$classe, p=.75, list=FALSE)
  training <- theData[inTrain,]
  testing <- theData[-inTrain,]
  
  return (list(training = training, testing = testing))
}
