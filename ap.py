from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


## 生成的测试数据的中心点
centers = [[1, 1], [-1, -1], [1, -1]]
##生成数据
Xn, labels_true = make_blobs(n_samples=150, centers=centers, cluster_std=0.5,
                            random_state=0)



simi = []
for m in Xn:
    ##每个数字与所有数字的相似度列表，即矩阵中的一行
    temp = []
    for n in Xn:
         ##采用负的欧式距离计算相似度
        s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(s)
    simi.append(temp)

p=-50   ##3个中心
#p = np.min(simi)  ##9个中心，
#p = np.median(simi)  ##13个中心

ap = AffinityPropagation(damping=0.5,max_iter=500,convergence_iter=30,
                         preference=p).fit(Xn)
cluster_centers_indices = ap.cluster_centers_indices_

for idx in cluster_centers_indices:
    print(Xn[idx])