# Setup
setwd("spr2017-proj3-group8/data")

library(gbm)
library(data.table)
library(dplyr)
library(caret)
source("../lib/gbm.R")

# Load data
sift <- fread("../output/extracted.pca 0.3 ")
sift<-fread("../output/hog_feature.csv")
sift<-fread("../output/sift_features/sift_features.csv",header=TRUE)
sift<-unlist(t(sift))


label <- read.table("../data/labels.csv",header=T)
label <- as.factor(unlist(label))
label_train<-label
dat_train<-sift


# Train the model and tune parameters
fit_train_gbm<-gbm_train(dat_train, label_train)


# Use the optimal model to fit the whole training data set and test the model 

tm_test <- system.time(pred_test <- gbm_test(fit_train_gbm, dat_test))

# Error rate
mean(pred_test!=label_train)