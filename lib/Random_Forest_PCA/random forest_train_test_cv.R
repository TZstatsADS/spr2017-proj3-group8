# Random Forest: Train
# choose between the method "ranger" and "randomForest"
# if ranger==TRUE, use ranger function in "ranger" package to train random forest
rf_train <- function(dat_train, label_train, ntree=500, 
                     saveFile=FALSE, ranger=TRUE){
  
  library(ranger)
  library(dplyr)
  #browser()
  train <- data.frame(dat_train) %>% mutate(label=factor(label_train))

  if (ranger==TRUE){
    fit <- ranger(label~.,
                data = train,
                num.trees = ntree)
  }else{
    fit <- randomForest::randomForest(label~.,
                  data = train,
                  ntree = ntree,
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
rf_cv <- function(dat_train, label_train, K=5, ntree=500){
  library(ranger)
  
  n <- length(label_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    cat(i/K) #processing record
    
    train.data <- dat_train[s != i,]
    train.label <- label_train[s != i]
    test.data <- dat_train[s == i,]
    test.label <- label_train[s == i]
    
    rf_fit <- rf_train(dat_train = train.data, label_train = train.label, ntree = ntree)
#browser()
    rf_predict <- rf_test(fit_train = rf_fit, dat_test = test.data)
    
    cv.error[i] <- mean(rf_predict != test.label)
    
  }
  
  error <- mean(cv.error)
  sd <- sd(cv.error)
  
  return(c(error, sd))
}


# Random Forest with PCA: Train
rf_pca_train <- function(dat_train, label_train, ntree=350, pca_threshold=0.2){
  
  pc_train <- feature.pca(dat_feature = dat_train, threshold = pca_threshold)

  fit_train_rf <- rf_train(dat_train = pc_train, label_train = label, ntree = ntree)
  
  return(fit_train_rf)
}



# Random Forest with PCA: Test
rf_pca_test <- function(fit_train, dat_test, ntree=350, pca_threshold=0.2){
  
  pc_test <- feature.pca(dat_feature = dat_test, threshold = pca_threshold)
  
  predict_rf <- rf_test(fit_train = fit_train, dat_test = pc_test)
  
  return(predict_rf)
}