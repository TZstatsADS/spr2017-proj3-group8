# Function: use PCA to extrac features
feature.pca <- function(dat_feature, threshold=0.85){
  
  # Run PCA on features
  feature.pca <- prcomp(as.data.frame(dat_feature), center = TRUE, scale. = TRUE)
  summary.pca <- summary(feature.pca)
  sd.pca <- summary.pca$sdev
  prop_var <- summary.pca$importance[2, ]
  cum_var <- summary.pca$importance[3,]
  
  # PCA threshold values
  thre <- which(cum_var >= threshold)[1]
  
  # PCA visualization
  png(filename=paste("../figs/pca visualization", threshold, ".png"))
  op <- par(mfrow=c(1,2))
  plot(seq(1,length(sd.pca), by=1), prop_var, type="l", 
       xlab = "PCA", ylab = "Proportion of variance",
       main = "Proportion of Variance")
  abline(h=threshold, col="red")
  abline(v=thre, col="blue")
  
  cum_var <- summary.pca$importance[3,]
  plot(seq(1,length(sd.pca), by=1), cum_var, type="l", 
       xlab = "PCA", ylab = "Cumulation of variance",
       main = "Cumulation of Variance")
  abline(h=threshold, col="red")
  abline(v=thre, col="blue")
  par(op)
  dev.off()
  
  
  # Extract first N PCAs based on threshold values
  pca_thre <- as.matrix(dat_feature) %*% feature.pca$rotation[,c(1:thre)]
  
  # save file
  save(pca_thre, file = paste("../output/extracted.pca", threshold, ".RData"))
  
  return(pca_thre)
}


