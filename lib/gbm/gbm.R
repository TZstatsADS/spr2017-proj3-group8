
gbm_train <- function(dat_train, label_train, ntree=250,depth=3){
  
  
  library("gbm")
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=ntree,
                     distribution="adaboost",
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  
  return(list(fit=fit_gbm, iter=best_iter))
}



gbm_test<-function(fit_train,dat_test) {
  library("gbm")
  pred_gbm<-predict(fit_train,newdata=dat_test,
                    n.trees=fit_train$n.trees,depth=fit_train$interaction.depth,
                    type="response")
  result<-as.numeric(pred_gbm>0.5)
 
  write.csv(result, file = "gbm_predict.csv")
 
  
  return(result)
}

gbm_cv<-function(X.train,y.train,K=5,depth=2){
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
    
    fit_gbm <- gbm.fit(x=train.data, y=train.label,
                       n.trees=250,
                       distribution="adaboost",
                       interaction.depth=depth, 
                       shrinkage=0.001,
                       bag.fraction = 0.5,
                       verbose=FALSE)
    
    best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
    
    pred_gbm<-predict(fit_gbm,newdata=test.data,n.trees=best_iter,type="response")
    pred<-as.numeric(pred_gbm>0.5)
    cv.error[i] <- mean(pred != test.label)
    
  }
  
  error <- mean(cv.error)
  sd <- sd(cv.error)

  return(c(mean(cv.error), sd(cv.error)))
  
  
}
  