# Load functions
source("NN_train_test_cv.R")

# Load features and label
library(data.table)
library(dplyr)
feature <- read.csv("../../output/hog_feature+sift.csv", header = TRUE)
feature_orig <- read.csv("../../output/sift_features/sift_features.csv", header = TRUE)
feature_orig <- as.data.frame(t(feature_orig))
label <- fread("../../data/labels.csv")
label <- c(t(label))

######### Tuning parameters #########

#### Ignore this section if optimal training parameter for hidden layers already known
#### hiddenLayers_origFeat <- 5
#### hiddenLayers_newFeat <- 3
#### As found in our tuning shown belown

# Tune parameter number of hidden layers

layers <- c(1,2,5,10,20)

cv <- vector("list", length(layers))
i <- 1

while (i < length(cv)) {
  cv[[i]] <- nn_cv(feature, label, K=5, hiddenLayers=layers[i])
  i = i+1
}




# Visualize CV results
q <- unlist(cv)
q2 <- q[c(TRUE,FALSE)]
png(filename=paste("../../figs/cv_result_nn.png"))
plot(y=q2, x=layers, type='l', xlab="Number of Neurons in Hidden Layer", ylab="5-Fold Avg CV Error", main="NN Parameter Tuning")
dev.off()

dev.print(png, "../../figs/cv_result_nn.png", width=500, height=400)

####
#### Begin here if known training parameter
####

# Choose the best parameter value from visualization

hiddenLayers_origFeat <- 5
hiddenLayers_newFeat <- 2

# train the model with the entire training set
fit_train_nn <- nn_train(train = feature, y = label, hiddenLayers = hiddenLayers_newFeat)
fit_train_nn_origFeat <- nn_train(train = feature_orig, y = label, hiddenLayers = hiddenLayers_origFeat)
save(fit_train_nn, file="../../output/fit_train_nn.RData")

# qq <- nn_cv(feature, label, K=5, hiddenLayers=2)
qq <- nn_cv(feature_orig, label, K=5, hiddenLayers=5)


### Make prediction 
# ?? fit_train_nn <- file("../../output/fit_train_nn.RData")
pred_test_nn <- nn_test(fit_train_nn, testData)
save(pred_test_nn, file="../../output/pred_test_nn.RData")

