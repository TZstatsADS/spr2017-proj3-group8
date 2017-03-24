# install packages
# --------------------------------------------------------------

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
install.packages("stringi")
install.packages("caret")

source("https://bioconductor.org/biocLite.R")
biocLite("EBImage", ask = F)

## pbapply is a library to add progress bar *apply functions
## pblapply will replace lapply
install.packages("pbapply")


# Load function to extract image features
# --------------------------------------------------------------
source("extract_feature.R")


# If not starting from scratch, load files from output
# --------------------------------------------------------------
load(file = "../../output/rawfeature_128X128.RData")
complete_set <- feature_matrix

## for 80% of data
load(file = "../../output/train_data.rda")
load(file = "../../output/test_data.rda")


# Load images as raw pixels
#-------------------------------------------------------------------------------
image_dir <- "../data/raw_images"

width <- 128
height <- 128

label <- read.csv("../data/labels.csv")
colnames(label) <- "label"
complete_set <- extract_feature(dir_path = image_dir, 
                                data_name = "128X128"
                                width = width, 
                                height = height)


# Split data into training and test and export files
#-------------------------------------------------------------------------------
library(caret)

## test/training partitions
training_index <- createDataPartition(complete_set$label, p = .8, times = 1)
training_index <- unlist(training_index)
train_set <- complete_set[training_index,]
dim(train_set)
train_data <- data.matrix(train_set)
save(train_data, file = "../output/train_data.rda")

test_set <- complete_set[-training_index,]
dim(test_set)
test_data <- data.matrix(test_set)
save(test_data, file = "../output/test_data.rda")

# Put data into format for model (train/test)
#-------------------------------------------------------------------------------
## Fix train and test datasets

train_x <- t(train_data[, -1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(width, height, 1, ncol(train_x))


test_x <- t(test_data[,-1])
test_y <- test_data[,1]
test_array <- test_x
dim(test_array) <- c(width, height, 1, ncol(test_x))


# Put data into format for model (FULL)
#-------------------------------------------------------------------------------

train_FULL_data <- complete_set
train_FULL_x <- t(train_FULL_data[, -1])
train_FULL_y <- train_FULL_data[,1]
train_FULL_array <- train_FULL_x
dim(train_FULL_array) <- c(width, height, 1, ncol(train_FULL_x))



# Set up the symbolic model
#-------------------------------------------------------------------------------

## start of train function
cnn_train <- function(train_x, train_y, output_model_name, saveFile=T, seed = 100,
                      num_round = 40, batch_size = 50, learn_rate = 0.01 
                      ) {
  library(mxnet)
  
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
  mx.set.seed(seed)
  
  # Device used. CPU in my case.
  devices <- mx.cpu()
  
  # Training
  #-------------------------------------------------------------------------------
  
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = train_x,
                                       y = train_y,
                                       ctx = devices,
                                       num.round = num_round,
                                       array.batch.size = batch_size,
                                       learning.rate = learn_rate,
                                       momentum = 0.9,
                                       wd = 0.00001,
                                       eval.metric = mx.metric.accuracy,
                                       epoch.end.callback = mx.callback.log.train.metric(100))
  if (saveFile == TRUE){
    mx.model.save(model, prefix = paste0("../../output/mxnet_", output_model_name), iteration = 1)
  }
}



# Test
#-------------------------------------------------------------------------------
## Predict on test set

cnn_test <- function(model, test_array, saveFile=FALSE) {
  predict_probs <- predict(model, test_array)
  predicted_labels <- max.col(t(predict_probs)) - 1
  predict_table <- table(test_data[, 1], predicted_labels)
    
  if (saveFile == TRUE){
    write.csv(predict_table, file = "../output/cnn_predict.csv")
  }
    
  ## accuracy rate
  accuracy <- sum(diag(table(test_data[, 1], predicted_labels)))/nrow(test_data)
  return(accuracy)
}

cnn_test(model, test_array, saveFile = T)


model <- mx.model.load(prefix = "../output/mxnet_model", iteration = 1)

