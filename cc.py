# Importing Modules
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


datafile = u'./data/k_input.xlsx'  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
d = DataFrame(data)
# Extract the measurements as a NumPy array
samples = d.values
mergings = linkage(samples, method='complete')
dendrogram(mergings,leaf_rotation=90,leaf_font_size=6,)
plt.show()