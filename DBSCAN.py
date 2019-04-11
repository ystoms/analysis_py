from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
from pandas import DataFrame


datafile = u'./data/k_input.xlsx'  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
d = DataFrame(data)

# Load Dataset
# Declaring Model
dbscan = DBSCAN()
# Fitting
dbscan.fit(d.values)
# Transoring Using PCA
pca = PCA(n_components=2).fit(d.values)
pca_2d = pca.transform(d.values)
# Plot based on Class
c1 = None
c2 = None
c3 = None
for i in range(0, pca_2d.shape[0]):
  if dbscan.labels_[i] == 0:
      c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
  elif dbscan.labels_[i] == 1:
      c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
  elif dbscan.labels_[i] == -1:
      c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

print(c1,c2,c3)
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
plt.title('DBSCAN finds 2 clusters and Noise')
plt.show()