import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


selected_cols = pd.read_csv('./gene500.csv')
x_data = pd.read_csv('./2k_fibro_update.csv')
y_data = pd.read_csv('./2k_fibro_labels.csv')

x_data = x_data.transpose()
print(x_data.shape)

selected_cols = selected_cols[selected_cols.gene500 != 'BAG3']
print('BAG3' in selected_cols['gene500'].values)
print('BAG3' in x_data.iloc[0, :].values)
gene_names = list(selected_cols.iloc[:,1].values)



x_data.columns = x_data.iloc[0,:]
print(x_data.shape)
x_data = x_data.iloc[1:,:]
print(x_data.shape)
x_data = x_data[selected_cols['gene500']]
print(x_data.shape)



x_data = x_data.to_numpy()
x_data = x_data.astype(np.float32)
y_data = y_data.to_numpy()
y_data = y_data[:,1]


print('start knn')
k = 100
###########################################################################################################
# import faiss
# index = faiss.IndexFlatL2(x_data.shape[1])
# index.add(np.ascontiguousarray(x_data))
#
# D, I = index.search(np.ascontiguousarray(x_data), x_data.shape[0])


from annoy import AnnoyIndex
t = AnnoyIndex(x_data.shape[1], 'dot')
for i in range(x_data.shape[0]):
    t.add_item(i, x_data[i,:])

t.build(25) # 10 trees

I = np.zeros((x_data.shape[0], x_data.shape[0]))
for i in range(x_data.shape[0]):
    I[i,:] = t.get_nns_by_item(i, x_data.shape[0])
#######################################################################################################################

F_idx = np.zeros((x_data.shape[0], k))
for i in range(x_data.shape[0]):
    y_i = y_data[I[i,:].astype(int)]
    if y_i[0] == 0:
        idx = np.where(y_i == 1)[0]
        idx = I[i,idx]
        F_idx[i,:] = idx[:k]
    if y_i[0] == 1:
        idx = np.where(y_i == 0)[0]
        idx = I[i,idx]
        F_idx[i,:] = idx[:k]


print('start building F')

F_idx = F_idx.astype(int)
F =[]
Y = np.log(x_data + 0.001)
for j in range(x_data.shape[1]):
    print(j)
    F.append(np.zeros((x_data.shape[0], k)))
    # Y.append(np.zeros((x_data.shape[0])))
    for i in range(x_data.shape[0]):
        # Y[j][i] = np.log(x_data[i,j]+0.001)
        F[j][i,:] = np.log(x_data[F_idx[i,:],j] + 1)

print('start linear regression')

from sklearn.linear_model import LinearRegression
import copy
beta = []
sigma = []
for j in range(x_data.shape[1]):
    print(j)
    beta.append(np.zeros(k))
    sigma.append(0)
    reg = LinearRegression().fit(F[j], Y[:, j])
    beta[j] = copy.deepcopy(reg.coef_)
    sigma[j] = copy.deepcopy(reg.intercept_)

print('start cf')

Y_cf = np.zeros(x_data.shape)
for j in range(x_data.shape[1]):
    Y_cf[:,j] = np.exp(np.matmul(F[j], beta[j]))


print('start ploting')
diff = (x_data - Y_cf)**2
diff = np.sum(diff, axis=0)
plt.xlim(0,diff.shape[0])
plt.stem(range(diff.shape[0]),diff)
plt.show()
sorted_idx = np.argsort(diff)
print(sorted_idx[-50:])
import pdb
pdb.set_trace()
print(" ".join(np.asarray(gene_names)[sorted_idx[-50:]]))
