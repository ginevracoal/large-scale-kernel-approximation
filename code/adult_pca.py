import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn import datasets

from functions import *
from encode_csv import *


n=20000
path='individual/code/'

adult_train = pd.read_csv(
    path+"../datasets/adult.data.txt",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

adult_train_encoded, encoders = number_encode_features(adult_train)

y_train = adult_train_encoded["Target"].values
# y_train = y_train.values

# scale between -1 and 1
X_train = adult_train_encoded.iloc[:,:-1]
X_train = scale_columns(X_train)


X = X_train[:n]
y = y_train[:n]

n_components=2

pca = PCA(n_components).fit(X)

X_pca = pca.transform(X)

X1, X2 = np.linspace(-0.5, 0.5, 50), np.linspace(-0.5, 0.5, 50)
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T


fig = tools.make_subplots(rows=2, cols=2,
                          subplot_titles=("Original space",
                                          "Projection by Kernel PCA",
                                          "Projection by Fourier PCA",
                                          "Projection by Nystroem PCA"))


reds = y == 0
blues = y == 1

original_space1 = go.Scatter(x=X.iloc[reds, 0], 
                             y=X.iloc[reds, 1],
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='red',
                                         line=dict(color='black', width=1))
                            )
original_space2 = go.Scatter(x=X.iloc[blues, 0],
                             y=X.iloc[blues, 1],
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='blue',
                                         line=dict(color='black', width=1))
                             
                            )

lines = go.Contour(x=X1, 
                   y=X2, 
#                    z=Z_grid, 
                   showscale=False,
                   colorscale=[[0,'white'],[1, 'black']],
                   contours=dict(coloring='lines')
                  )

fig.append_trace(lines, 1, 1)
fig.append_trace(original_space1, 1, 1)
fig.append_trace(original_space2, 1, 1)

fig['layout']['xaxis1'].update(title='x<sub>1</sub>',
                               zeroline=False, showgrid=False)
fig['layout']['yaxis1'].update(title='x<sub>2</sub>',
                               zeroline=False, showgrid=False)


# kernel pca


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=False, gamma=10)
X_kpca = kpca.fit_transform(X)

projection_kpca1 = go.Scatter(x=X_kpca[reds, 0], 
                              y=X_kpca[reds, 1],
                              mode='markers',
                              showlegend=False,
                              marker=dict(color='red',
                                          line=dict(color='black', width=1)) 
                             )
projection_kpca2 = go.Scatter(x=X_kpca[blues, 0], 
                              y=X_kpca[blues, 1], 
                              mode='markers',
                              showlegend=False,
                              marker=dict(color='blue',
                                          line=dict(color='black', width=1))  
                             )

fig.append_trace(projection_kpca1, 2, 1)
fig.append_trace(projection_kpca2, 2, 1)

fig['layout']['xaxis3'].update(title="1st principal component",
                               zeroline=False, showgrid=False)
fig['layout']['yaxis3'].update(title='2nd component',
                               zeroline=False, showgrid=False)


## Fourier

fourier = RBFSampler(gamma=gamma, random_state=1)

# X_pca = pca.fit_transform(X)
fourier.fit(X_pca)
X_fourier_pca = fourier.transform(X_pca)

projection_fourier_1 = go.Scatter(x=X_fourier_pca[reds, 0], 
                             y=X_fourier_pca[reds, 1], 
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='red',
                                         line=dict(color='black', width=1)) 
                            )
projection_fourier_2 = go.Scatter(x=X_fourier_pca[blues, 0], 
                             y=X_fourier_pca[blues, 1], 
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='blue',
                                         line=dict(color='black', width=1))
                            )

fig.append_trace(projection_fourier_1, 1, 2)
fig.append_trace(projection_fourier_2, 1, 2)

fig['layout']['xaxis2'].update(title='1st principal component',
                               zeroline=False, showgrid=False)
fig['layout']['yaxis2'].update(title='2nd component',
                               zeroline=False, showgrid=False)


## Nystroem

nystroem = Nystroem(gamma=gamma, random_state=1)
nystroem.fit(X_pca)
X_nystroem_pca = nystroem.transform(X_pca)

projection_nystroem_1 = go.Scatter(x=X_nystroem_pca[reds, 0], 
                             y=X_nystroem_pca[reds, 1], 
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='red',
                                         line=dict(color='black', width=1)) 
                            )
projection_nystroem_2 = go.Scatter(x=X_nystroem_pca[blues, 0], 
                             y=X_nystroem_pca[blues, 1], 
                             mode='markers',
                             showlegend=False,
                             marker=dict(color='blue',
                                         line=dict(color='black', width=1))
                            )

fig.append_trace(projection_nystroem_1, 2, 2)
fig.append_trace(projection_nystroem_2, 2, 2)

fig['layout']['xaxis2'].update(title='1st principal component',
                               zeroline=False, showgrid=False)
fig['layout']['yaxis2'].update(title='2nd component',
                               zeroline=False, showgrid=False)


py.iplot(fig)

plotly.offline.plot(fig, filename='adult_pca.html') 