install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(mxnet)

# Load images
#-------------------------------------------------------------------------------
# image_dir <- "../data/raw_images"
image_dir <- "../train"


source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")
library(EBImage)

## pbapply is a library to add progress bar *apply functions
## pblapply will replace lapply
install.packages("pbapply")
library(pbapply)
extract_feature <- function(dir_path, width, height, is_cat = TRUE, add_label = TRUE) {
  img_size <- width*height
  ## List images in path
  images_names <- list.files(dir_path)
  
  ### comment out later
  if (add_label) {
    ## Select only cats or dogs images
    images_names <- images_names[grepl(ifelse(is_cat, "cat", "dog"), images_names)]
    ## Set label, cat = 0, dog = 1
    label <- ifelse(is_cat, 0, 1)
  }  
  #####################
  
  
  print(paste("Start processing", length(images_names), "images"))
  
  
  ## This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    
    
    ### COMMENT OUT LATER??
    ## Set to grayscale
    grayimg <- channel(img_resized, "gray")
    #######################
    
    ## Get the image as a matrix
    img_matrix <- grayimg@.Data
    ## Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if (add_label) {
    ## Add label
    feature_matrix <- cbind(label = label, feature_matrix)
  }
  return(feature_matrix)
}


width <- 72
height <- 72
cats_data <- extract_feature(dir_path = image_dir, width = width, height = height)
dogs_data <- extract_feature(dir_path = image_dir, width = width, height = height, is_cat = FALSE)

dim(cats_data)
dim(dogs_data)

saveRDS(cats_data, "cat2.rds")
saveRDS(dogs_data, "dog2.rds")

# # Load data(sift_features) and label
# #-------------------------------------------------------------------------------
# install.packages("data.table")
# install.packages("dplyr")
# library(data.table)
# library(dplyr)
# feature <- fread("../output/sift_features/sift_features.csv", header = TRUE)
# label <- fread("../data/labels.csv")
# label <- c(t(label))
# feature <- tbl_df(t(feature)) 
# complete_set <- cbind(label, feature)

# df <- extract_feature(dir_path = image_dir, width = width, height = height)
# complete_set <- df


# Split data into training and test
#-------------------------------------------------------------------------------
install.packages("caret")
library(caret)
## Bind rows in a single dataset
complete_set <- rbind(cats_data, dogs_data)

## test/training partitions
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
dim(train_array) <- c(72, 72, 1, ncol(train_x))

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(72, 72, 1, ncol(test_x))


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
mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 60,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
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
