# Based on tuning result, best depth is 3
gbm_train<-function(X.train, y.train, n.tree=250,depth=3){
  
  fit.model<-gbm.fit(x=X.train, y=y.train,
                     n.trees=n.tree,
                     distribution='bernoulli',
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  
  return(fit.model)
  
}

gbm_train_cv <- function(dat_train, label_train){
  

  library("gbm")
  depth_values<-c(1,3,5,7,9)
  err_cv<-matrix(NA,nrow=length(depth_values), ncol=2)
  
  K=5
  for(k in 1:length(depth_values)){
    err_cv[k,] <- gbm_cv(dat_train, label_train, K=K,depth=depth_values[k])
  }
  
  save(err_cv, file="../output/err_cv.csv")
  
  depth_best <- depth_values[which.min(err_cv[,1])]
  
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=250,
                     distribution="adaboost",
                     interaction.depth=depth_best, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  fit_train<-list(fit=fit_gbm, iter=best_iter)
  return(list(fit=fit_gbm, iter=best_iter,depth=depth_best))
}


gbm_test<-function(fit_train,dat_test) {
  library("gbm")
  pred_gbm<-predict(fit_train,newdata=dat_test,
                    n.trees=fit_train$iter,depth=fit_train$depth,type="response")
  result<-as.numeric(pred_gbm>0.5)
  if (saveFile == TRUE){
  write.csv(result, file = "gbm_predict.csv")
   }
  
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
  