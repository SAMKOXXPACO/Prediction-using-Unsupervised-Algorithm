#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[1]:


# Lets choose K-Means Clustering Unsupervised ML Algorithm


# In[3]:


# Step 1: Let us import the required Libraries


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns


# In[5]:


# Step-2: Let us load the Iris Data set from sklearn


# In[6]:


iris=datasets.load_iris()


# In[7]:


iris


# In[8]:


iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)


# In[9]:


iris_df


# In[10]:


# Step3: Let us do a Exploratory Data Analysis for the Iris Data Set
#Displays the first 5 rows of the dataset
iris_df.head() 


# In[11]:


#Displays the dimensions of the dataset
iris_df.shape


# In[12]:


#Displays the numerical insights on the dataset
iris_df.describe() 


# In[13]:


#Checking for null values in the dataset
iris_df.isnull().sum()


# In[14]:


# Step 4: Visualize and Analyze the Dataset


# In[15]:


#Finding the corelation between the data

corr_df= iris_df.corr()
corr_df


# In[16]:


#Plotting a heat map for the dataset

plt.figure(figsize= [10,6])
sns.heatmap(corr_df, cmap='Spectral', annot=True)


# In[17]:


#PLotting a graph by considering different attributes in pairs

sns.pairplot(iris_df)


# In[18]:


# Step 5: Design the K-Means Clustering Algorithm for optimal clusters


# In[19]:


#Extracting the values of different attributes in the dataset such as sepal lenth, sepal width, petal length and petal width
x = iris_df.iloc[:, [0, 1, 2, 3]].values


# In[20]:


x


# In[ ]:


# Step-6: We actually do not know the number of clusters. 
#There are several methods to select k that depends on the domain knowledge and rule of thumbs. 
# Elbow method is one of the robust one used to find out the optimal number of clusters.
#In this method, the sum of distances of observations from their cluster centroids, called Within-Cluster-Sum-of-Squares (WCSS).
#This is computed as the shown where Yi is centroid for observation Xi


# ![image.png](attachment:image.png)

# In[97]:


#KMeans class from the sklearn library.
# Using the elbow method to find out the optimal number of #clusters.

# Now we will difine the K means clustering algorithm. As we do not know what is the optimum number of clusters.
# The way to do this is using FOR loop by keeping the range from 1 to 10 since we dont want large number of clusters depiction
# We want to find what is the optimum number of clusters 
# would be storing the value of each iterations in the list called WCSS(Within Cluster Sum of Squares) and we are using that to
# plot our graph
# Within Cluster Sum of Squares (WCSS)

#i above is between 1-10 numbers. init parameter is the random #initialization method  
#we select kmeans++ method. max_iter parameter the maximum number of iterations there can be to 
#find the final clusters when the K-meands algorithm is running. we #enter the default value of 300
#the next parameter is n_init which is the number of times the #K_means algorithm will be run with
#different initial centroid.
# K-Means Clustering algorithm to find optimal clusters for classification
    
#kmeans algorithm fits to the X dataset
 #appending the WCSS to the list (kmeans.inertia_ returns the WCSS value for an initialized cluster)


# In[98]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
   kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
   kmeans.fit(X)
   #appending the WCSS to the list (kmeans.inertia_ returns the WCSS value for an initialized cluster)
   wcss.append(kmeans.inertia_)


# In[99]:


# Step 7: Plot the K-Means Clustering graph and identify the optimal number of clusters from the graph.


# In[100]:


# kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
# Plotting the results onto a line graph, allowing us to observe 'The elbow'

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


plt.plot(range(1,11), wcss)
plt.title('The elbow method to find optimal number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[103]:


#From 'The Elbow Method' of graphical representation, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.

#Therefore, from the above graph we choose the optimal number of clusters to be 3.


# In[110]:


# Step 8 
# Applying kmeans to the dataset / Creating the kmeans classifier with optimal clusters to be 3 and than fitting the model to 
# do the predictions
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[111]:


y_kmeans


# In[112]:


# Step 9: Visualize the Clusters using a scatter plot


# In[115]:


# Visualising the clusters on the first two columns

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 80, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 80, c = 'pink', label = 'Iris-virginica')

# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 200, c = 'black', label = 'Centroids')

plt.legend()


# In[116]:


# Step 6: Make Predictions


# In[117]:


#Defining the input and target variables

X = iris.data[:,:2]     #Contains sepal length and Sepal Width
y = iris.target         #Contains target species value


# In[118]:


X


# In[119]:


y


# In[120]:


# Visualizing X and Y variables in graphical form

plt.scatter(X[:,0],X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Sepal Length', fontsize=14)
plt.ylabel('Sepal Width', fontsize=14)
plt.show()


# In[121]:


# Step 7: Evaluate the Model: Comparing Actual vs Predicted data values


# In[122]:


#This will tell us which cluster the data observation belongs to

new_labels = kmeans.labels_
new_labels


# In[123]:


#Plotting the identified clusters and comparing with the results

fig, axes = plt.subplots(1,2, figsize=(16,8))
axes[0].scatter(X[:,0],X[:,1], c=y, cmap='gist_rainbow', edgecolor = 'k',s=80)
axes[1].scatter(X[:,0],X[:,1], c=new_labels, cmap='viridis', edgecolor = 'k',s=80)


axes[0].set_xlabel('Speal Length',fontsize=18)
axes[0].set_ylabel('Speal Width',fontsize=18)

axes[1].set_xlabel('Speal Length',fontsize=18)
axes[1].set_ylabel('Speal Width',fontsize=18)

axes[0].tick_params(direction='in',length=10,width=5,colors='k',labelsize=20)
axes[1].tick_params(direction='in',length=10,width=5,colors='k',labelsize=20)

axes[0].set_title('Actual',fontsize=18)
axes[1].set_title('Predicted',fontsize=18)

plt.show()


# In[44]:


# 3-D Plotting
# K means Clustering


# In[124]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
X = iris.data
#KMeans
km = KMeans(n_clusters=3)
km.fit(X)
km.predict(X)
labels = km.labels_
#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)


# In[125]:


# 3-D Plotting
# Gaussian Mixture Model


# In[126]:


from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
X = iris.data
#Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
proba_lists = gmm.predict_proba(X)
#Plotting
colored_arrays = np.matrix(proba_lists)
colored_tuples = [tuple(i.tolist()[0]) for i in colored_arrays]
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=colored_tuples, edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Gaussian Mixture Model", fontsize=14)


# In[ ]:




