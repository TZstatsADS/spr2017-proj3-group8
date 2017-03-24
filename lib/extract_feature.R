#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Original Author: Shikun Li
### Sourced from: https://rpubs.com/kanedglsk/236125
### Adapted by: Ken Chew

extract_feature <- function(dir_path, width, height, 
                            data_name="data", is_cat = TRUE, 
                            add_label = TRUE, export = T) {
  

  library(EBImage)
  
  ## pbapply is a library to add progress bar *apply functions
  ## pblapply will replace lapply
  library(pbapply)
  
  img_size <- width*height
  
  ## List images in path
  images_names <- list.files(dir_path)
  
  
  print(paste("Start processing", length(images_names), "images"))
  
  
  ## This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    ## Set to grayscale
    grayimg <- channel(img_resized, "gray")
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
  
  ### output constructed features
  if(export){
    save(feature_matrix, file=paste0("../output/rawfeature_", data_name, ".RData"))
  }
  return(feature_matrix)
}