# Load functions
source("NN_train_test_cv.R")


# Load features and label
library(data.table)
library(dplyr)
feature <- read.csv("../../output/hog_feature+sift_resize.csv", header = TRUE)
rownames(feature) <- feature$X
feature <- subset(feature, select=-c(X))
label <- fread("../../data/labels.csv")
label <- c(t(label))

######### Tuning parameters #########

# Tune parameter number of hidden layers

layers <- c(1,2,5,10,20,40,100)

cv <- vector("list", length(layers))
i <- 1
impr <- TRUE

while (i < length(cv)) {
  cv[[i]] <- nn_cv(feature, label, K=5, hiddenLayers=layers[i])
  i = i+1
}

q <- unlist(cv)
q2 <- q[c(TRUE,FALSE)]
plot(q2, type='l')


# Visualize CV results
q <- unlist(cv)
q2 <- q[c(TRUE,FALSE)]
plot(y=q2, x=layers[1:6], type='l', xlab="Number of Neurons in Hidden Layer", ylab="5-Fold Avg CV Error")
png(filename=paste("../../figs/cv_result_nn.png"))
dev.off()

dev.print(png, "../../figs/cv_result_nn.png", width=500, height=400)

# Choose the best parameter value from visualization

hiddenLayers_origFeat <- 5
hiddenLayers_newFeat <- 3

# train the model with the entire training set
fit_train_nn <- nn_train(train = feature, y = label, hiddenLayers = hiddenLayers_newFeat)
save(fit_train_nn, file="../../output/fit_train_nn.RData")

### Make prediction 
# ?? fit_train_nn <- file("../../output/fit_train_nn.RData")
pred_test_nn <- nn_test(fit_train_nn, testData)
save(pred_test_nn, file="../../output/pred_test_nn.RData")

