from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn import datasets

##################################################

# default parameters
gamma=0.1
C = 100
scale_samples = 10

# SVM classifiers
kernel_svm = svm.SVC(kernel='rbf', gamma=gamma, C=C)
linear_svm = svm.SVC(kernel='linear', C=C)

# the two methods
random_fourier = RBFSampler(gamma=gamma, random_state=1)
nystroem = Nystroem(gamma=gamma, random_state=1)

# pipelines for kernel approximations
fourier_svm = pipeline.Pipeline([("feature_map", random_fourier),("svm", linear_svm)])
nystroem_svm = pipeline.Pipeline([("feature_map", nystroem),("svm", linear_svm)])\

##################################################


def pca_projections(train_data, train_labels, n_components=1):

		if n_components == 1:
				n_components = train_data.shape[1]//2

		## calculate real and predicted projections of points on the plane

		# visualize the decision surface, projected down to the first
		# principal components of the dataset
		pca = PCA(n_components=n_components).fit(train_data)

		X = pca.transform(train_data)

		# Generate grid along first two principal components
		multiples = np.arange(-2, 2, 0.1)
		# steps along first component
		first = multiples[:, np.newaxis] * pca.components_[0, :]
		# steps along second component
		second = multiples[:, np.newaxis] * pca.components_[1, :]
		# combine
		grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
		flat_grid = grid.reshape(-1, train_data.shape[1])

		# title for the plots
		titles = ['SVC with rbf kernel',
		          'SVC (linear kernel)\n with Fourier rbf feature map\n'
		          'n_components='+str(n_components),
		          'SVC (linear kernel)\n with Nystroem rbf feature map\n'
		          'n_components='+str(n_components)]

		plt.tight_layout()
		plt.figure(figsize=(12, 5))

		fourier_svm.fit(train_data, train_labels)
		nystroem_svm.fit(train_data, train_labels)
		kernel_svm.fit(train_data, train_labels)


		for i, clf in enumerate((kernel_svm, 
		                         fourier_svm,
		                         nystroem_svm)):
		    # Plot the decision boundary. For that, we will assign a color to each
		    # point in the mesh [x_min, x_max]x[y_min, y_max].
		    plt.subplot(1, 3, i + 1)
		    Z = clf.predict(flat_grid)

		    # Put the result into a color plot
		    Z = Z.reshape(grid.shape[:-1])
		    plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
		    plt.axis('off')

		    # Plot also the training points
		    plt.scatter(X[:, 0], X[:, 1], c=train_labels, cmap=plt.cm.Paired,
		                edgecolors=(0, 0, 0))

		    plt.title(titles[i])

		# plt.tight_layout()
		plt.show()
        


