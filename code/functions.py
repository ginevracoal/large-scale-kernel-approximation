import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time

from sklearn import svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn import datasets

from keras.datasets import mnist
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# default parameters
gamma=0.1
C = 100
scale_samples = 30

# SVM classifiers
kernel_svm = svm.SVC(kernel='rbf', gamma=gamma, C=C)
linear_svm = svm.SVC(kernel='linear', C=C)

# the two methods
random_fourier = RBFSampler(gamma=gamma, random_state=1)
nystroem = Nystroem(gamma=gamma, random_state=1)

# pipelines for kernel approximations
fourier_svm = pipeline.Pipeline([("feature_map", random_fourier),("svm", linear_svm)])
nystroem_svm = pipeline.Pipeline([("feature_map", nystroem),("svm", linear_svm)])

# number of random samples
def samples(train_data, scale_samples=scale_samples):
		# samples = len(train_data)//60 * np.arange(1,10)
		samples = scale_samples * np.arange(4,11)
		return(samples)


# fit and predict linear and kernel SVMs
def kernel(train_data, train_labels, test_data, test_labels, gamma=gamma, C=C):
		print("\nkernel svm fitting")
		start = time()
		kernel_svm.fit(train_data, train_labels)
		kernel_svm_score = kernel_svm.score(test_data,test_labels)
		kernel_svm_time = time() - start

		return({'score':kernel_svm_score, 'time':kernel_svm_time})

def linear(train_data, train_labels, test_data, test_labels, C=C):
		print("\nlinear svm fitting")
		start = time()
		linear_svm.fit(train_data, train_labels)
		linear_svm_score = linear_svm.score(test_data, test_labels)
		linear_svm_time = time() - start
		return ({'score':linear_svm_score, 'time':linear_svm_time})

def nystroem(train_data, train_labels, test_data, test_labels, scale_samples=scale_samples, gamma=gamma, random_state=1):
		print("\nnystroem svm fitting")

		nystroem_scores = []
		nystroem_times = []
		sample_sizes = samples(train_data, scale_samples)

		for D in sample_sizes:
				print("\n", D, "/", max(sample_sizes),"samples")
				# set the number of samples
				nystroem_svm.set_params(feature_map__n_components=D)

				start = time()
				nystroem_svm.fit(train_data, train_labels)
				nystroem_times.append(time() - start)

				nystroem_score = nystroem_svm.score(test_data, test_labels)
				nystroem_scores.append(nystroem_score)
		
		return({'scores':nystroem_scores, 'times':nystroem_times})

def fourier(train_data, train_labels, test_data, test_labels, scale_samples=scale_samples, gamma=gamma, random_state=1):

		print("\nfourier svm fitting")

		fourier_scores = []
		fourier_times = []
		sample_sizes = samples(train_data, scale_samples)

		for D in sample_sizes:
				print("\n", D, "/", max(sample_sizes),"samples")
				fourier_svm.set_params(feature_map__n_components=D)

				start = time()
				fourier_svm.fit(train_data, train_labels)
				fourier_times.append(time() - start)

				fourier_score = fourier_svm.score(test_data, test_labels)
				fourier_scores.append(fourier_score)
		
		return({'scores':fourier_scores, 'times':fourier_times})


# fits all methods and predicts test set
def fit_all(train_data, train_labels, test_data, test_labels, scale_samples=30, gamma=gamma, C=C, random_state=1):
	  
		print("\nusing gamma=", gamma)
		sample_sizes = samples(train_data, scale_samples)
		nystroem_out = nystroem(train_data, train_labels, test_data, test_labels, scale_samples, gamma, random_state)
		fourier_out = fourier(train_data, train_labels, test_data, test_labels, scale_samples, gamma, random_state)
		kernel_out = kernel(train_data, train_labels, test_data, test_labels, gamma=gamma, C=C)
		linear_out = linear(train_data, train_labels, test_data, test_labels, C=C)

		return({'kernel':kernel_out, 'linear':linear_out, 'nystroem':nystroem_out, 'fourier':fourier_out, 'sample_sizes':sample_sizes})

def save(fit, filename):
		abs_path = '/galileo/home/userexternal/gcarbone/individual/code/pickle/'
		with open(abs_path+filename+'.pickle', 'wb') as handle:
				pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

# takes a fit_all argument


def plot_results(fit):

		sample_sizes = fit['sample_sizes']

		kernel = fit['kernel']
		linear = fit['linear']
		nystroem = fit['nystroem']
		fourier = fit['fourier']

		kernel_svm_score = kernel['score']
		linear_svm_score = linear['score']
		nystroem_scores = nystroem['scores']
		fourier_scores = fourier['scores']

		kernel_svm_time = kernel['time']
		linear_svm_time = linear['time']
		nystroem_times = nystroem['times']
		fourier_times = fourier['times']
		  
		# plot the results
		plt.figure(figsize=(8, 8))

		accuracy = plt.subplot(211)
		# second y axis for timings

		accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")

		accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
		timescale.plot(sample_sizes, fourier_times, '--',
		               label='Fourier approx. kernel')

		# horizontal lines for exact rbf and linear kernels:
		accuracy.plot([sample_sizes[0], sample_sizes[-1]],
		              [linear_svm_score, linear_svm_score], label="linear svm")
		timescale.plot([sample_sizes[0], sample_sizes[-1]],
		               [linear_svm_time, linear_svm_time], '--', label='linear svm')

		accuracy.plot([sample_sizes[0], sample_sizes[-1]],
		              [kernel_svm_score, kernel_svm_score], label="RBF svm")
		timescale.plot([sample_sizes[0], sample_sizes[-1]],
		               [kernel_svm_time, kernel_svm_time], '--', label='RBF svm')

		# vertical line for dataset dimensionality = 64
		# accuracy.plot([64, 64], [0.7, 1], label="n_features")

		# legends and labels
		# accuracy.set_title("Classification accuracy")
		# timescale.set_title("Training times")
		accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
		timescale.set_xlim(sample_sizes[0], sample_sizes[-1])
		# accuracy.set_xticks(())
		accuracy.set_ylim(np.min(fourier_scores), kernel_svm_score*1.001)
		timescale.set_xlabel("Number of samples")
		accuracy.set_xlabel("Number of samples")
		accuracy.set_ylabel("Classification accuracy")
		timescale.set_ylabel("Training time (min)")
		accuracy.legend(loc='best')
		timescale.legend(loc='best')

		plt.show()

def plot_accuracy(fit):

		sample_sizes = fit['sample_sizes']

		kernel = fit['kernel']
		linear = fit['linear']
		nystroem = fit['nystroem']
		fourier = fit['fourier']

		kernel_svm_score = kernel['score']
		linear_svm_score = linear['score']
		nystroem_scores = nystroem['scores']
		fourier_scores = fourier['scores']
		  
		# plot the results
		plt.figure(figsize=(8, 8))

		accuracy = plt.subplot(211)

		accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
		accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
		# horizontal lines for exact rbf and linear kernels:
		accuracy.plot([sample_sizes[0], sample_sizes[-1]],
		              [linear_svm_score, linear_svm_score], label="linear svm")
		accuracy.plot([sample_sizes[0], sample_sizes[-1]],
		              [kernel_svm_score, kernel_svm_score], label="RBF svm")



		# vertical line for dataset dimensionality = 64
		# accuracy.plot([64, 64], [0.7, 1], label="n_features")

		# legends and labels
		# accuracy.set_title("Classification accuracy")
		# timescale.set_title("Training times")
		accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])

		# accuracy.set_xticks(())
		accuracy.set_ylim(np.min(fourier_scores), kernel_svm_score*1.02)
		accuracy.set_xlabel("Number of samples")
		accuracy.set_ylabel("Classification accuracy")
		accuracy.legend(loc='best')



		plt.show()

def plot_timescale(fit):

		sample_sizes = fit['sample_sizes']

		kernel = fit['kernel']
		linear = fit['linear']
		nystroem = fit['nystroem']
		fourier = fit['fourier']

		kernel_svm_time = kernel['time']*60
		linear_svm_time = linear['time']*60
		nystroem_times = [time*60 for time in nystroem['times']]
		fourier_times = [time*60 for time in fourier['times']]	
		  
		# plot the results
		plt.figure(figsize=(8, 8))

		timescale = plt.subplot(212)

		timescale.plot(sample_sizes, nystroem_times, '--', label='Nystroem approx. kernel')
		timescale.plot(sample_sizes, fourier_times, '--', label='Fourier approx. kernel')
		timescale.plot([sample_sizes[0], sample_sizes[-1]],
		               [linear_svm_time, linear_svm_time], '--', label='linear svm')
		timescale.plot([sample_sizes[0], sample_sizes[-1]],
		               [kernel_svm_time, kernel_svm_time], '--', label='RBF svm')
		timescale.set_xlim(sample_sizes[0], sample_sizes[-1])
		timescale.set_xlabel("Number of samples")
		timescale.set_ylabel("Training time (min)")
		timescale.legend(loc='best')