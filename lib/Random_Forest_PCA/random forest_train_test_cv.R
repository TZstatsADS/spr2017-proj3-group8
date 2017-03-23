# Random Forest: Train
# choose between the method "ranger" and "randomForest"
# if ranger==TRUE, use ranger function in "ranger" package to train random forest
rf_train <- function(dat_train, label_train, ntree=500, mtry=sqrt(ncol(dat_train)), 
                     saveFile=FALSE, ranger=TRUE){
  
  library(ranger)
  library(dplyr)
  
  train <- data.frame(dat_train) %>% mutate(label=factor(label_train))
  
  if (ranger==TRUE){
    fit <- ranger(label~.,
                data = train,
                num.trees = ntree,
                mtry = mtry)
  }else{
    fit <- randomForest::randomForest(label~.,
                  data = train,
                  ntree = ntree,
                  mtry = mtry,
                  type = "classification")
  }
  
  if (saveFile == TRUE){
    save(file = "../output/fitted_rf_model.RData", rf_fit)
  }
  
  return(fit)
}



# Random Forest: Test
rf_test <- function(fit_train, dat_test, saveFile=FALSE){
  
  rf_predict <- predict(fit_train, dat_test)$predictions
  
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
  
  for (i in 1:K){
    cat(i/K) #processing record
    
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

#rf_cv(X.train=feature, y.train=label, K=5) ## return c(0.28450000 0.02706474)
