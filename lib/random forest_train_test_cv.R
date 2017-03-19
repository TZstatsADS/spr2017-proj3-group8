
library(data.table)
library(dplyr)

# set working directory here
setwd('E:/statistics/applied data science/Project3/spr2017-proj3-group8/lib') ##better way?

# Load data(sift_features) and label
feature <- fread("../output/sift_features/sift_features.csv", header = TRUE)
label <- fread("../data/labels.csv")
label <- c(t(label))
feature <- tbl_df(t(feature))

# Random Forest: Train
rf_train <- function(dat_train, label_train, ntree=500, mtry=sqrt(ncol(dat_train)), saveFile=FALSE){
  
  library(randomForest)
  library(dplyr)
  
  train <- data.frame(dat_train) %>% mutate(label=factor(label_train))
  fit <- randomForest(label~.,
                      data = train,
                      ntree = ntree,
                      mtry = mtry,
                      type = "classification")
  
  if (saveFile == TRUE){
    save(file = "../output/fitted_rf_model.RData", rf_fit)
  }
  
  return(fit)
}



# Random Forest: Test
rf_test <- function(fit_train, dat_test, saveFile=FALSE){
  
  rf_predict <- predict(fit_train, dat_test)
  
  if (saveFile == TRUE){
    write.csv(rf_predict, file = "../output/rf_predict.csv")
  }
  
  return(rf_predict)
}



# Random Forest with cross-validation
rf_cv <- function(X.train, y.train, K=5, ntree=500, mtry=sqrt(ncol(X.train))){
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  print("Sampling completed")
  
  for (i in 1:K){
    print(i/K) #processing record
    
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
  
    rf_fit <- rf_train(dat_train = train.data, label_train = train.label, ntree = ntree, mtry = mtry)
    rf_predict <- rf_test(fit_train = rf_fit, dat_test = test.data)
    
    cv.error[i] <- mean(rf_predict != test.label)
    
  }
  
  error <- mean(cv.error)
  sd <- sd(cv.error)
  
  return(c(error, sd))
}

# rf_cv(X.train=feature, y.train=label, K=5) ## return c(0.28200000 0.02723738)
