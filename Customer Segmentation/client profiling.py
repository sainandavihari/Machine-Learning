import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


data= pd.read_csv(r"/Users/sainandaviharim/Desktop/Files/Python Projects/ML Projects/Customer Segmentation/Mall_Customers.csv")

data.columns

X=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# Here , Inertia means centroid calculation.

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# From the graph,We can observed that , we need to create 5 clusters for this dataset.
# So , If we build model with 5 clusters , then it would be a good model.

kmeans1=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans1.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


data['prediction rating']=y_kmeans

data.head()

output='Mall_Customers_Final.csv'
data.to_csv(output,index=False)

full_path = os.path.abspath(output)
print(full_path)

filename = 'kmeans1.pkl'
with open(filename, 'wb') as file:
    pickle.dump(kmeans1, file)