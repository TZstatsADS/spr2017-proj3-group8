######################################################
######### Overall Project Train Script ###############
######################################################

# load functions from the models
source("../lib/gbm/gbm.R")
source("../lib/Neural_Net/NN_train_test_cv.R")
source("../lib/Random_Forest_PCA/random forest_train_test_cv.R")
source("../lib/Conv_Neural_Net/mxnet_train_test_cv.R")


overall_train <- function(dat_train, cnn_train_data, label_train) {
  fit_train_gbm <- gbm_train(dat_train, label_train)
  fit_train_nn <- nn_train(train = dat_train, y = label_train)
  fit_train_rf <- rf_train(dat_train, label_train)
  cnn_train(train_x = cnn_train_data, train_y = label_train, output_model_name = "FinalTrain")
  
  save(fit_train_gbm, fit_train_nn, fit_train_rf, file = "trained_models.RData")
}

gbm_train(dat_train, label_train)
