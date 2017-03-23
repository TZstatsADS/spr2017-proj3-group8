# Setup
setwd("../data")

library(gbm)
library(data.table)
library(dplyr)
library(caret)
source("../lib/GBM.R")

# Load data
sift <- fread("../output/sift_features/sift_features.csv", header = TRUE)
# load(paste("../output/extracted.pca", 0.3, ".RData"))
# sift<-pca_thre

sift <- unlist(t(sift)) 

label <- read.table("../data/labels.csv",header=T)
label <- as.factor(unlist(label))
label_train<-label
dat_train<-sift


# Train the model and tune parameters
fit_train_gbm<-gbm_train(data_train, label_train)

# 

# Test the model 

