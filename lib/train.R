######################################################
######### Overall Project Train Script ###############
######################################################

# load functions from the models
source("gbm/gbm.R")
source("Neural_Net/NN_train_test_cv.R")
source("Random_Forest_PCA/random forest_train_test_cv.R")
source("Conv_Neural_Net/mxnet_train_test_cv.R")


overall_train <- function(dat_train, cnn_train_data, label_train) {
  fit_train_gbm <- gbm_train(dat_train, label_train)
  fit_train_nn <- nn_train(train = dat_train, y = label_train)
  fit_train_rf <- rf_pca_train(dat_train, label_train)
  fit_train_cnn <- cnn_train(train_x = cnn_train_data, train_y = label_train)
}