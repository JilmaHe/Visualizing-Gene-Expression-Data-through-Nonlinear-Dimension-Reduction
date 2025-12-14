A0<- read.table("C:/Users/Jilma/Desktop/TCGA.BRCA.sampleMap_AgilentG4502A_07_3",header = T)
#library(readxl)
#library(dplyr)
#library(openxlsx)
A<-na.omit(A0)

A<-A[,-1:-4]
A<-as.matrix(A)

summary(as.numeric(A))
# Compute k-means clustering for different values of k
wcss <- numeric(15)  #within-cluster sum of squares
for (i in 1:10) {
  kmeans_result <- kmeans(A, centers=i)
  wcss[i] <- kmeans_result$tot.withinss
}
# Plot the elbow curve
plot(1:15, wcss, type='b', pch=19, frame=FALSE, 
     xlab='Number of clusters', ylab='WCSS', main='Elbow Method')
# Add a line at the "elbow" point
segments(x0=11, y0=min(wcss), x1=11, y1=max(wcss), lty=2, col='red')

