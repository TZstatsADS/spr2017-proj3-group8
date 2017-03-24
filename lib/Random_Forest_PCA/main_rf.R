# Set directory


# Load functions
source("random forest_train_test_cv.R")


# Load features and label
library(data.table)
library(dplyr)
feature <- fread("../../output/hog_feature+sift.csv", header = TRUE)
label <- fread("../../data/labels.csv")
label <- c(t(label))
feature <- tbl_df(t(feature)) 


######### Tuning parameters #########

# Tune parameter for random forest: ntree
ntree <- seq(10, 400, by=20) 


err_cv_rf <- c()
err_sd_rf <- c()

  
for (j in 1:length(ntree)){
  cat("j=", j, "\n")
  result <- rf_cv(dat_train = pca_thre, label_train = label, K = 5, ntree = ntree[j])
  err_cv_rf[j] <- result[1]
  err_sd_rf[j] <- result[2]
}
  

# Save results
save(err_cv_rf, file="../../output/err_cv_rf.RData")
save(err_sd_rf, file="../../output/err_sd_rf.RData") 

# Visualize CV results
png(filename=paste("../../figs/cv_result_rf.png"))
plot(x=ntree, y=err_cv_rf, type="l", ylab="error rate",main="Random Forest")
dev.off()

# Choose the best parameter value from visualization
best_ntree <- 350


############# Retrain model with tuned parameters ##############

# train the model with the entire training set
tm_train_rf <- system.time(fit_train_rf <- rf_train(dat_train=feature, label_train=label, ntree=best_ntree))
save(fit_train_rf, file="../../output/fit_train_rf.RData")


### Make prediction 
tm_test_rf <- system.time(pred_test_rf <- rf_test(fit_train = fit_train_rf, dat_test = pc_test))
save(pred_test_rf, file="../../output/pred_test_rf.RData")




