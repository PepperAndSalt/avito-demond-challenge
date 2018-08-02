$$a=b$$
# My First Kaggle Competition

I just finished my first Kaggle competition. My goal was to place in the top 10%. In the end, I placed at 83rd place (top 5%) and won a silver medal. I am writing this notebook to summarize my first experience 

## The challenge

The challenge was hosted by Avito.au, a Russian classified advertisements website. Avito is the most popular classified site in Russian and is the third biggest site in the world after Craigslist and 58.com. The challenge was to predict demand for online advertisement based on itf full description(title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. With this information, Avito can inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive.

## My solution 

I think this competition is very good and also very challenge for a Kaggle beginner. Except some common numerical and catigorical features, this competition also has lots of text and image features which make this competition very challenge. In this section, I will conclude my solution here. 

## Feature Engineering

### Geographic feature

In the dataset, it provides information about the city and region an ad was placed in. So at first, my idea was to extact latitude and longtitude informations. I parsed these information by using google map API. I also parsed cities' population from wikipedia. Then I was about trying to group cities into different clusters. I tried to use Kmeans but as [this kernel](https://www.kaggle.com/frankherfert/region-and-city-details-with-lat-lon-and-clusters/notebook) said a kmeans cluster will not do much good here because it will create equally sized area clusters, which doesn't reflect the real world distribution of cities.

Note: This part are mainly from this [website](http://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) (click the link for details)

**HDBSCAN**- Hierarchical Density-Based Spatial Clustering of Applications with Noise. 
Regular DBScan is amazing at clustering data of varying shapes, but falls short of clustering data of varying density. HDBSCAN performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more robust to parameter selection. 

The core of the clustering algorithm is single linkage clustering, and it can be quite sensitive to noise. Obviously we want our algorithm to be robust against noise so we need to find a way to help ‘lower the sea level’ before running a single linkage algorithm.

First, let's define some metrics. Core distance $core_k(x)$ is defined as the distance between point x and the kth nearest point:
\begin{align*}
core_k(x) = d(x, N^k(x))
\end{align*}

Mutual reachability distance is defined as 
\begin{align*}
$d_{mreach-k}(a,b) = max\{core_k(a), core_k(b), d(a,b)\}$
\end{align*}
where $d(a,b)$ is the original metric distance between $a$ and $b$. 

In general there is underlying theory to demonstrate that mutual reachability distance as a transform works well in allowing single linkage clustering to more closely approximate the hierarchy of level sets of whatever true density distribution our points were sampled from.

(hierarchical clustering identifies groups in a tree-like structure but
suffers from computational complexity in large datasets while K-means clustering is efficient but designed to
identify homogeneous spherically-shaped clusters.)

#### Build the minimum spanning tree

We can build the minimum spanning tree very efficiently via Prim’s algorithm – we build the tree one edge at a time, always adding the lowest weight edge that connects the current tree to a vertex not yet in the tree.

#### Build the cluster hierarchy

Given the minimal spanning tree, the next step is to convert that into the hierarchy of connected components. This is most easily done in the reverse order: sort the edges of the tree by distance (in increasing order) and then iterate through, creating a new merged cluster for each edge.

Then, we can do cluster extraction. The first step in cluster extraction is condensing down the large and complicated cluster hierarchy into a smaller tree with a little more data attached to each node. 

To make this concrete we need a notion of **minimum cluster size** which we take as a parameter to HDBSCAN. We can now walk through the hierarchy and at each split ask if one of the new clusters created by the split has fewer points than the minimum cluster size. 

1. If we have fewer points than the minimum cluster size we declare it to be ‘points falling out of a cluster’ and have the larger cluster retain the cluster identity of the parent. 
2. If on the other hand the split is into two clusters each at least as large as the minimum cluster size then we consider that a true cluster split and let that split persist in the tree

#### Extract the clusters

To make a flat clustering we will need to add a further requirement that, if you select a cluster, then you cannot select any cluster that is a descendant of it.

First we need a different measure than distance to consider the persistence of clusters; instead we will use $\lambda = \frac{1}{\mathrm{distance}}$.

Define $\lambda_{\mathrm{birth}}$ and $\lambda_{\mathrm{death}}$ to be the lambda value when the cluster split off and became it’s own cluster, and the lambda value (if any) when the cluster split into smaller clusters respectively. 

Define $\lambda_p$ as the lambda value at which that point ‘fell out of the cluster’ which is a value somewhere between $\lambda_{\mathrm{birth}}$ and $\lambda_{\mathrm{death}}$. Now, for each cluster compute the stability as
\begin{align*}
\sum_{p \in \mathrm{cluster}} (\lambda_p - \lambda_{\mathrm{birth}}).
\end{align*}

Now work up through the tree (the reverse topological sort order). 
1. If the sum of the stabilities of the child clusters is greater than the stability of the cluster, then we set the cluster stability to be the sum of the child stabilities. 
2. If, the cluster’s stability is greater than the sum of its children then we declare the cluster to be a selected cluster and unselect all its descendants. 

Once we reach the root node we call the current set of selected clusters our flat clustering and return that.

### Image Features

There are 1.5 million images. According to this great [kernel](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality), I calculated dullness, whiteness, average pixel width, key colos, dimensions and blurrness. 

#### Average pixel width

Some images may contain no pixel variation and are entirely uniform. Average Pixel Width is a measure which indicates the amount of edges present in the image. If this number comes out to be very low, then the image is most likely a uniform image and may not represent right content.

Skimage's Canny Detection can perform the edge detection. The Canny filter is a multi-stage edge detector. It uses a filter based on the derivative of a Gaussian in order to compute the intensity of the gradients.The Gaussian reduces the effect of noise present in the image. Then, potential edges are thinned down to 1-pixel curves by removing non-maximum pixels of the gradient magnitude. Finally, edge pixels are kept or removed using hysteresis thresholding on the gradient magnitude.

#### Image blurriness

In this [blog](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/), it uses variation of the Laplacian by Pech-Pacheco et al. in their 2000 ICPR paper, [Diatom autofocusing in brightfield microscopy: a comparative study](http://optica.csic.es/papers/icpr2k.pdf) to calcualte image blurriness. 

You simply take a single channel of an image and convolve it with the following 3 x 3 kernel:

$ M = \begin{bmatrix} 0 & 1 & 0 
\\ 1 & -4 & 1
\\ 0 & 1 & 0 \end{bmatrix} $

And then take the variance (i.e. standard deviation squared) of the response. If the variance falls below a pre-defined threshold, then the image is considered blurry; otherwise, the image is not blurry.


The reason this method works is due to the definition of the Laplacian operator itself, which is used to measure the 2nd derivative of an image. The Laplacian highlights regions of an image containing rapid intensity changes, much like the Sobel and Scharr operators. And, just like these operators, the Laplacian is often used for edge detection. The assumption here is that if an image contains high variance then there is a wide spread of responses, both edge-like and non-edge like, representative of a normal, in-focus image. But if there is very low variance, then there is a tiny spread of responses, indicating there are very little edges in the image. As we know, the more an image is blurred, the less edges there are.

### Object Detection

The objects in the image surely have effect to the prediction. I used [YOLOv3](https://pjreddie.com/darknet/yolo/) to perform object detection to identify the firstt tree objects in the image. We can use pretrained model to get the job done. It took a while to finish all 1.5 million images. A python wrapper is [here](https://github.com/AlexeyAB/darknet)

### Text Feature

TF-IDF and SVD. Embedding vectors using fastText

### Other Feature Engineering

#### Date time feature

Date time features are not very important in this competition since there are only two month of data. I only extract weekday from activation date as a date time feature

#### Feature interactions

There are some catigorical features which we can combine them together. image_top_1 appears to be the most important features but why? One of the guesses is the relativeness to the ads category, which can be represented by the count of ads of each image class (image_top_1) within a category. Price is another important feature therefore we may want to create more interactions between price and other categorical features with higher cardinality, e.g. region_city_combined_category, which provices the lowest granularity of categories.

#### Mean Encoding

There are lots of high cardinality features. So I use mean encoding to encode these high cardinality categorical features. Mean encoding is an effective aproach to encode categorical features by processing train targets (y) with Empirical Bayes which works best with high-cardinality categorical features. However, it is prone to overfitting thus needs to be carefully tuned. As a result, it did improve my score a lot. 

The following explanation of mean encoding is from [this paper](http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.pdf).

Assume target $y$ has $C$ different classes, a certain feature $X$ has $K$ different values, each value is represented as $k$. Then.
1. Prior probability is $P(y = target)$
2. Posterior probability is $P(y = target | X = k)$

The basic idea of this algorithm is to encode each value k to be its estimated posterior probability:
\begin{align*}
\hat{P} (target = y | variable = k)= \frac{number\ of\ y=target\ \&\ X=k}{number\ of\ X=k}
\end{align*}

But, the more typical senario for a high-cardinality categorical feature is one where the number of unique values of $X$ is very large and the records are unevenly distributed across the possible values of $X$. In this scenario many of the cells will be characterized by a small smaple size ($number\ of\ X=k$), therefore the direct estimat of posterior probability would be highly unreliable.

To mitigate the effect of small cell counts the probability estimates are calculated as the mixture of two probabilities as 
\begin{align*}
\hat{P} = \lambda * \hat{P}(y=target) + (1-\lambda)*\hat{P} (target = y | variable = k)
\end{align*}

where $\hat{P}(y=target)=(number\ of\ y=target)\ /\ (number\ of\ y)$

$\lambda(n) = \frac{1}{1+e^{(n-k)/f}}$

Also note that, if you use all the data to fit and use all the data to do transform, then it will be overfitting. Therefore, we need to split the data in to $n$ folds. Each fold is encoded by using the rest $n-1$ folds data. The larger the $n$, the more accurate its encoding, but also more memory consumption and more computation time.

Mean Encoding python implementation can be found [here](https://zhuanlan.zhihu.com/p/26308272)

## Modelling

I trained one XGBoost, two LightGBM, one RNN and a ridge regression. I used a 5-fold CV followed by stacking with a LightGBM. My RNN is simple. A one-layer LSTM was concatenate with other features and then passed through 2 Dense layers. I was about to train two NN models. But because of my poor machine (8GB), I could only train a baseline model on my local computer. And I was first training my models on a spot instance on AWS. But I got terminated twice in the middle of the training. So sad.... Then I switched to a regular instance and paid regular price. So I didn't have enough time train my second NN model.

Finally, thanks again to people who have shared their kernels and methods so generously on Kaggle. There are still a lot of rooms I should improve and lots of things to learn. I indeed enjoy my first Kaggle journey. At last, Happy Kaggling!
