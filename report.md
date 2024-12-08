# Subquestion 1: Classification with the Iris Dataset

In this subquestion, we trained a knn classifier on the iris dataset to be able to predict a flower's species.
We decided to evaluate the knn model with the default hyperparameters first, then if they do not perform satisfactorily, we could finetune the model.
Luckily, we didn't need to finetune it as it yielded the following results with default parameters:
```
species  precision  f1_score  recall
      0     1.0000  1.000000  1.0000
      1     0.9375  0.967742  1.0000
      2     1.0000  0.967742  0.9375
Accuracy: 0.98
```

We also decided to plot our data using a matrix scatter plot to visualize the relationship between every two features.
From said plot, it can very clearly be seen that there are distinct clusters in the data, especially as seen by the relationship between petal width x sepal width and and petal width x petal length.

![Iris dataset matrix scatterplot](https://imgur.com/yPiBMaX.png)

# Subquestion 2: Simple Linear Regression with a Synthetic Dataset

Here we started by creating a synthetic linear dataset and adding random normal noise to it with a standard deviation of 1.5.
We then plot the data. We use the function provided in the project prompt with a slope of 3.
Our model predicts the data with an MSE of 2.16 with a slope of 3.01.

![Linear regression plot with points and regression line](https://imgur.com/u9pKJ08.png)

# Subquestion 3: Clustering with a Synthetic Dataset

In this subquestion, the goal was to create 3 distinct clusters artificially, then create a kmeans model and use that to predict the clusters for each point in our artificial dataset, then plot the points and clusters.
We decided to use random normal distributions for generating the clusters, with the following parameters for the three clusters:
```python
# first two elements of the tuples are cluster center's x and y,
# second point is spread
cluster_xyspread_list = [
    (2, 2, 1.5),
    (10, 10, 2.5),
    (1, 12, 2)
]
```

We decided to use 500 points for each cluster, for a total of 1500 points.

The plot seems to get the clusters right, but due to the large spread and small distances between clusters, a few points
are roughly equally distanced to more than one cluster center, making it slightly unclear which cluster they belong
to. This can be tweaked by modifying the cluster centers and spread above in the code, if desired.

![Scatterplot of cluster points and center](https://imgur.com/BrUJX9I.png)

# Subquestion 4: Time Series Analysis with Synthetic Data

This subquestion dealt with generating a synthetic time series dataset that is mostly linear but has a seasonal component that is a sine wave, in addition to some random noise following a random distribution.

The code allows for the customization of many parameters, using the following variables at the top of the file:
```python
n_points = 500 # 500 days
linear_slope = 0.05
seasonal_scale = 5
noise_weight = 1
```

We then train a linear regression model on our time series data, and plotted both the points and the linear model's slope line.
We can see that the linear model has the same overall direction as the time series data; however, it does not meet it perfectly due to its being a linear model while our data is not exactly linear, mostly due to the inclusion of the sine wave in the generation of the synthetic dataset.
We can see that the error decreases significantly as the `seasonal_scale` variable decreases.

![Synthetic time series data with linear model plot](https://imgur.com/rJaPc9c.png)