# esempio qui: http://peekaboo-vision.blogspot.com/2012/12/kernel-approximations-for-efficient.html

# import functions
import encode_csv
from functions import * 
from encode_csv import * 


path='/galileo/home/userexternal/gcarbone/individual/'

covtype = pd.read_csv(path+'datasets/covtype.data')

X = covtype.drop(['5'], axis=1) 
le = LabelEncoder().fit(covtype['5']) 
y = le.transform(covtype['5'])

# encode data
encoded, encoders = number_encode_features(X)

# scale columns between -1 and 1
X = scale_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print("train:", len(X_train), ", test:", len(X_test))

# taking subsets

n = 600 
covtype_fit_600 = fit_all(X_train[:n], y_train[:n], X_test[:n//20], y_test[:n//20])
save(covtype_fit_600, 'covtype_fit_600')


