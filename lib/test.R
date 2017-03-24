### Source models

source("../lib/Neural_Net/NN_train_test_cv.R")
source("../lib/gbm/gbm.R")
source("../lib/Random_Forest_PCA/random forest_train_test_cv.R")


test <- function(dat_test){
  library(data.table)
  library(dplyr)
  library(ranger)
  library(neuralnet)
  library(gbm)
  
  #Load trained baseline model (JUST SIFT FEATURES) with known parameters
  load("../output/fit_train_gbm.RData")
  gbm_model <- fit_train_gbm$fit
  fit_train_gbm <- NA
  
  #Load trained models (new features) with known parameters
  load("../output/fit_train_rf.RData")
  rf_model_hs <- fit_train_rf
  load("../output/fit_train_gbm_hog.RData")
  gbm_model_hs <- fit_train_gbm_hog$fit
  nn_model_hs <- nn_train(train=feature_hs, y=label, hiddenLayers=2)
  
  #Test model (Baseline)
  preds_s <- gbm_test(fit_train=gbm_model, dat_test=dat_test)
  
  #Test Model (improved model, including hog features)
  rf_preds  <- as.numeric(rf_test(fit_train=rf_model_hs, dat_test=dat_test))-1
  nn_preds  <- as.numeric(nn_test(nn=nn_model_hs, test=dat_test))
  gbm_preds <- as.numeric(gbm_test(fit_train=gbm_model_hs, dat_test=dat_test))
  
  preds_hs <- rbind(rf_preds,nn_preds,gbm_preds)
  preds_hs <- round(colMeans(preds_hs),0)
  
  #Save predictions to dataframe provided and export
  testLabels$`Baseline (0 for chicken, 1 for dog)` <- preds_s
  testLabels$Advanced <- preds_hs
  
  write.csv(testLabels, "../output/labels.csv")
}
