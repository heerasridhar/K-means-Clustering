# K-means-Clustering and Hungarian algorithm

There are two datasets ATNT-face-image400 and Hand-written-26-letters.

data set: ATNT-face-image400.txt :

1st row is cluster labels. 2nd-end rows: each column is a feature vectors (vector length=28x23).

Total 40 classes. each class has 10 images. Total 40*10=400 images

data set: Hand-written-26-letters.txt :

Text file. 1st row is cluster labels. 2nd-end rows: each column is a feature vectors (vector length=20x16). Total 26 classes and each class has 39 images. Total 26*39=1014 images.

On these two data sets K-means clustering and Hungarian algorithm were used to predict the accuracy of the data sets.
The confusion matrix was generated and column-index were then rearranged and the confusion matrix be dispalyed.
