######################################################
######### Overall Project Train Script ###############
######################################################

# load functions from the models
source("../lib/gbm/gbm.R")
source("../lib/Neural_Net/NN_train_test_cv.R")
source("../lib/Random_Forest_PCA/random forest_train_test_cv.R")
source("../lib/Conv_Neural_Net/mxnet_train_test_cv.R")

load("trained_models.RData")
model <- mx.model.load(prefix = "../output/mxnet_FULL_model", iteration = 1)

overall_test <- function(dat_test, cnn_test_data) {
  pred_gbm <- gbm_test(fit_train_gbm$fit, dat_test)
  pred_nn <- nn_test(fit_train_nn, dat_test)
  pred_rf <- rf_test(fit_train_rf, dat_test)
  pred_cnn <- cnn_test(model, cnn_test_data)
  

  # accuracy of gbm, nn, rf, cnn
  sum <- .098 + .135 + .14 + .17
  accuracy <- c(.098, .135, .14, .17)
  accuracy <- accuracy / sum
  
  results <- data.frame(gbm = pred_gbm,
                nn = pred_nn,
                rf = pred_rf,
                cnn = pred_cnn)
  
  final_pred <- rowSums(t(t(results) * accuracy))
  return(as.numeric(final_pred> 0.5))
}
