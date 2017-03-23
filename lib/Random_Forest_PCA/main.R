# Set directory
setw

# Load functions
source("../lib/random forest_train_test_cv.R")
source("../lib/PCA.R")


# Load data(sift_features) and label
library(data.table)
library(dplyr)
feature <- fread("../output/sift_features/sift_features.csv", header = TRUE)
label <- fread("../data/labels.csv")
label <- c(t(label))
feature <- tbl_df(t(feature)) 

# Tune parameter of PCA: threshold
threshold_value <- seq(0.1, 0.9, by=0.1)

# If haven't saved pca results in output folder before:
# execute the following lines:
for (i in 1:length(threshold_value)){
  pca_thre <- feature.pca(dat_feature = feature, threshold = threshold_value[i])
}


# tune parameter for random forest: ntree
ntree <- seq(10, 400, by=10) 

err_cv <- matrix(NA, ncol=length(ntree), nrow=length(threshold_value))
err_sd <- matrix(NA, ncol=length(ntree), nrow=length(threshold_value))

for (i in 1:length(threshold_value)){
  cat("i=", i, "\n")
  threshold <- threshold_value[i]
  
  # Use the already saved extracted pca features with threshold value
  load(paste("../output/extracted.pca", threshold, ".RData"))
  
  for (j in 1:length(ntree)){
    cat("j=", j, "\n")
    result <- rf_cv(X.train = pca_thre, y.train = label, K = 5, ntree = ntree[j])
    err_cv[i,j] <- result[1]
    err_sd[i,j] <- result[2]
  }
}  

err_cv_df <- data.frame(err_cv)
colnames(err_cv_df) <- ntree
rownames(err_cv_df) <- threshold_value

err_sd_df <- data.frame(err_sd) 
colnames(err_sd_df) <- ntree
rownames(err_sd_df) <- threshold_value

library(tidyr)
library(ggplot2)
cleaned_err_cv <- err_cv_df %>% 
  tibble::rownames_to_column("threshold_value") %>% 
  gather(key = ntree, value = val, -threshold_value)

cleaned_err_sd <- err_sd_df %>% 
  tibble::rownames_to_column("threshold_value") %>% 
  gather(key = ntree, value = val, -threshold_value)

save(err_cv, err_cv_df, cleaned_err_cv, file="../output/err_cv_rf.RData")
save(err_sd, err_sd_df, cleaned_err_sd, file="../output/err_sd_rf.RData") 

# Visualize CV results
png(filename=paste("../figs/cv_result_rf.png"))
ggplot(cleaned_err_cv) +
  geom_line(aes(x=as.numeric(ntree), y=val, color=threshold_value)) +
  ggtitle("CV Results of Random Forest") +
  xlab("ntree")+
  ylab("error rate")
dev.off()
  
# Choose the best parameter value from visualization
best_pca_thre <- 0.4
best_ntree <- 300
  
# train the model with the entire training set
load(paste("../output/extracted.pca", best_pca_thre, ".RData"))
tm_train <- system.time(fit_train <- rf_train(dat_train = pca_thre, label_train = label, ntree = best_ntree))
save(fit_train, file="../output/fit_train_rf.RData")




