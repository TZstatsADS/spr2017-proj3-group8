# Neural Network: Train

nn_train <- function(train, y, hiddenLayers=5, saveFile=FALSE){
  
  library(dplyr)
  library(neuralnet)
  
  #features <- as.data.frame(scale(train, center=TRUE, scale=TRUE))
  features <- train
  n <- names(features)
  train_c <- cbind(features,y)
  
  ## For rows with all zero, center around 0 and remain after scaling
  #for (column in colnames(train_c)[colSums(is.na(train_c)) > 0]){
  #  print(column)
  #  train_c[column] <- rep(0, nrow(train_c))
  #}
  
  f <- as.formula(paste("y ~", paste(n[!n %in% "features"], collapse = " + ")))

  fit <- neuralnet::neuralnet(f, data=train_c, hidden=hiddenLayers, linear.output=FALSE)
  
  if (saveFile == TRUE){
    save(file = "../output/fitted_nn_model.RData", fit)
  }
  
  return(fit)
}



# Neural Network: Test
nn_test <- function(nn, test, saveFile=FALSE){

  nn_predict <- neuralnet::compute(nn,test)
  nn_predict <- round(nn_predict$net.result,0)
  
  if (saveFile == TRUE){
    write.csv(nn_predict, file = "../output/nn_predict.csv")
  }
  
  return(nn_predict)
}



# Neural Network with cross-validation
nn_cv <- function(dat_train, label_train, K=5, hiddenLayers=5){
  
  n <- length(label_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  print(hiddenLayers)
  for (i in 1:K){
    cat(i/K) #processing record
    
    train.data <- dat_train[s != i,]
    train.label <- label_train[s != i]
    test.data <- dat_train[s == i,]
    test.label <- label_train[s == i]
    
    
    nn_fit <- nn_train(train=train.data, y=train.label, hiddenLayers = hiddenLayers)
    nn_predict <- nn_test(nn = nn_fit, test = test.data)
    
    cv.error[i] <- mean(nn_predict != test.label)
    
  }
  
  error <- mean(cv.error)
  sd <- sd(cv.error)
  
  return(c(error, sd))
}