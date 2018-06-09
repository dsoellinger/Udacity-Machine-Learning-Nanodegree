# Unsupervised Learning

## Why do we need unsupervised learning?

Unsupervised learning becomes important once we have data that don't have a label.

The idea is to come up with algorithms that are able to detect structures in a dataset.  
For example, we might wanna try to find the clusters in the dataset.

<img src="images/clustering_example.png" width="250"/>


## K-Means Clustering

K-Means is an algorithm that allows us to detect clusters in dataset. A cluster can be seen as a group of points that have something in common. This typically means that the points are close to each other.

Algorithm:

1. Randomly pick n arbitary points as cluster centers.
2. Now let's compute the distance from every data point to each cluster center.  
3. Assign each point to the closest cluster center.
4. We now try to optimize the positions of the cluster centers. Therefore, we **minimize** to total of the **quadratic distances** between each point of a cluster center and the cluster center.
5. Move the cluster centers.

The cost function looks as follows:

$J = \sum_{i=1}^k \sum_{x_{j} \in S_{i}} || x_{j} - \mu_{i} ||^2$

**Note:** Solving the problem is not trivial (NP-hard!). Therefore, k-means algorithm only hopes to find the global minimum and possibly gets stuck at a local minimum.


**Visualization:** https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

### Stability of k-means

**Will the output of K-Means always be the same?**

No, the result depends on the initial position of the cluster centers. Therefore, we can say that k-means **is not stable**.

Let's consider the following example:

<img src="images/clustering_bad_cluster_centers.png" width="300"/>

As we can see initializing k-means with the points shown in the image results in a very strange partitioning. Definitely, a partitioning we don't want.  
However, if we initialize the points differently, we might get another partitioning with cluster centers on the left/right.

There is no guarantee that k-means converges to the "right" solution.