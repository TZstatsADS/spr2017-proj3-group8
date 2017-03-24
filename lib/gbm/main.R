# Setup
setwd("spr2017-proj3-group8/data")

library(gbm)
library(data.table)
library(dplyr)
source("../lib/gbm.R")

# Load data

sift_hog<-fread("../output/hog_feature+sift.csv")
sift<-fread("../output/sift_features/sift_features.csv",header=TRUE)
sift<-unlist(t(sift))


label <- read.table("../data/labels.csv",header=T)
label <- unlist(label)
label_train<-label
dat_train<-sift
hog_train<-sift_hog

# Train the model and tune parameters
library("gbm")
depth_values<-c(1,3,5,7,9)
err_cv<-matrix(nrow=length(depth_values), ncol=2)

K=5
for(k in 1:length(depth_values)){
  err_cv[k,] <- gbm_cv(dat_train, label_train, K=K,depth=depth_values[k])
}

write.csv(err_cv, file="../output/err_cv_gbm.csv")

plot(depth_values, err_cv[,1], xlab="Interaction Depth", ylab="CV Error",
     main="Cross Validation Error",type="l",ylim=c(0,0.4))

depth_best1 <- depth_values[which.min(err_cv[,1])]

# best depth is 5
fit_train_gbm<-gbm_train(dat_train, label_train,depth=depth_best1)
pred_test1<-as.numeric(gbm_test(fit_train_gbm,dat_train)>0.5)


########################################################
# Train the model and tune parameters with hog features
err_cv_hog<-matrix(nrow=length(depth_values), ncol=2)

K=5
for(k in 1:length(depth_values)){
  err_cv_hog[k,] <- gbm_cv(hog_train, label_train, K=K,depth=depth_values[k])
}



write.csv(err_cv_hog , file="../output/err_cv_gbm_hog.csv")

plot(depth_values, err_cv[,1], xlab="Interaction Depth", ylab="CV Error",
     main="Cross Validation Error",type="l",ylim=c(0,0.4))

depth_best2 <- depth_values[which.min(err_cv[,1])]
# best depth = 9

# Use the optimal model to fit the whole training data set and test the model 


fit_train_gbm_hog<-gbm_train(hog_train, label_train,depth=depth_best2)
pred_test2<-as.numeric(gbm_test(fit_train_gbm_hog,hog_train)>0.5)

# Error rate
mean(pred_test1!=label_train)
mean(pred_test2!=label_train)
# 0.2355 & 0.098