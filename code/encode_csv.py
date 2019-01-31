# encode categorical featuers as numbers
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np

def number_encode_features(df):
  labels = df.copy()
  encoders = {}
  for column in labels.columns:
      if labels.dtypes[column] == np.object:
          encoders[column] = LabelEncoder()
          labels[column] = encoders[column].fit_transform(labels[column])
  return labels, encoders

def scale_columns(csv):
  min_max_scaler = preprocessing.MinMaxScaler()
  
  for column in csv.columns:    
      col = csv[column]
      csv[column] = (col-col.mean(axis=0))/(max(col)-min(col))
  return csv