# Setup
setwd("../data")
used.packages<-c("gbm","data.table","dplyr","caret","e1071")
library(gbm)
library(data.table)
library(dplyr)
library(caret)


# Load data
sift <- fread("C:/Users/yj2360/Documents/project3/project3/spr2017-proj3-group8/output/sift_features/sift_features.csv", header = TRUE)

sift <- data.frame(t(sift)) 

label <- read.table("labels.csv",header=T)
label <- c(t(label))
label_train<-label
dat_train<-sift

gbm_train(sift,label)
# Train the model and tune parameters





##################################################
# train.R
# tune parameter: n.tree & shrinkage & depth
# ntree = best iter, generated automatically.. no need to be tuned
# so, tune shrinkage & depth
gbm_train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  ### tuning is included
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  if(is.null(par)){
    depth <- c(1,2,3)
  } else {
    depth <- par$depth
  }
  
  # Find best parameters using cross validation: shrinkage + tree depth
  gbmGrid <-  expand.grid(interaction.depth = depth, 
  # Since the gbm package tunes the number of trees for fixed values of the tree depth and shrinkage.
                          n.trees=250,
                         # n.trees=(1:10)*100,
                          shrinkage = 0.001,
                          n.minobsinnode = 10)
  
  fitControl <- trainControl( method = "repeatedcv",
                              number = 10,
                              repeats = 5)
  set.seed(825)
  fit_gbm <- train(x=dat_train, y=label_train,
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  ## Now specify the exact models 
                  ## to evaluate:
                  tuneGrid = gbmGrid)
  
  paras<-fit_gbm$bestTune
  
  plot(fit_gbm)
  fit <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=paras$n.trees,
                     distribution="adaboost",
                     interaction.depth=paras$interaction.depth, 
                     shrinkage=paras$shrinkage,
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit, method="OOB",plot.it = FALSE)
  
  return(list(fit=fit, iter=best_iter))
}

###############################################
# test.R
test<-function(fit_train,dat_test) {
  library("gbm")
  pred_gbm<-predict(fit_train,newdata=dat_test,
                    n.trees=fit_train$iter,type="response")
  result<-as.numeric(pred_gbm>0.5)
  if (saveFile == TRUE){
    write.csv(result, file = "../output/gbm_predict.csv")
  }
  return(result)
}


###############################################
# feature.R
# On a new set of images and SIFT descriptors, 
# each team will have 30 minutes to process them into features chosen.
# Submit the processed features as a folder of feature objects file.
# [https://github.com/TZstatsADS/Fall2016-proj3-grp10/blob/master/lib/SIFTtry.R]