# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
# Loading dataset

datafile = u'./data/k_input.xlsx'  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
d = DataFrame(data)
# Defining Model
model = TSNE(learning_rate=100)
# Fitting Model
transformed = model.fit_transform(d.values)
# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]
plt.scatter(x_axis, y_axis)
plt.show()