#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Yifei Lin
### Project 3
### ADS Spring 2017


HOG_features <- function(dir_image,dir_sift,set_name) {
  
  ### Load libraries
  if(!require(EBImage)){
    source("http://bioconductor.org/biocLite.R")
    biocLite("EBImage")
  }
  if(!require(OpenImageR)){
    install.packages("OpenImageR")
  }
  library(OpenImageR)
  library(EBImage)
  
  
  ### Extract 448 HOG features
  dir_names <- list.files(dir_image)
  HOG_df <- data.frame(matrix(nrow = length(dir_names), ncol = 448))
  for(i in 1:length(dir_names)) {
    HOG_df[i,]<-HOG(readImage(paste0(dir_image,"/",dir_names[i])),cells = 8,orientations = 7)
  } 
  
  
  ### Combine hog features with sift features
  sift <- read.csv(dir_sift)
  sift <- as.data.frame(t(sift))
  final_features <- cbind(sift,HOG_df)
  
  
  ### output Constructed Features
  write.csv(final_features, file = paste0("./output/feature_",set_name,".csv"),row.names = F)

  
  return(final_features)
  
}
