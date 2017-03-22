install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library("mxnet")


# Load data(sift_features) and label
#-------------------------------------------------------------------------------
install.packages("data.table")
install.packages("dplyr")
library(data.table)
library(dplyr)
feature <- fread("../output/sift_features/sift_features.csv", header = TRUE)
label <- fread("../data/labels.csv")
label <- c(t(label))
feature <- tbl_df(t(feature)) 
complete_set <- cbind(label, feature)


# Split data into training and test
#-------------------------------------------------------------------------------
install.packages("caret")
library(caret)
## test/training partitions
set.seed(1)
training_index <- createDataPartition(complete_set$label, p = .8, times = 1)
training_index <- unlist(training_index)
train_set <- complete_set[training_index,]
dim(train_set)
test_set <- complete_set[-training_index,]
dim(test_set)


# Put data into format for model
#-------------------------------------------------------------------------------
## Fix train and test datasets
train_data <- data.matrix(train_set)
train_x <- t(train_data[, -1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(1, 5000, 1, ncol(train_x))

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(1, 5000, 1, ncol(test_x))


# Set up the symbolic model
#-------------------------------------------------------------------------------

data <- mx.symbol.Variable('data')

# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(1)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 5,
                                     array.batch.size = 40,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))


# Test
#-------------------------------------------------------------------------------
## Test test set
predict_probs <- predict(model, test_array)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test_data[, 1], predicted_labels)

## accuracy rate
sum(diag(table(test_data[, 1], predicted_labels)))/nrow(test_set)
