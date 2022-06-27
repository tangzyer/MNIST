from utils import generate_gaussian_data
from utils import generate_subset_test_data
from gmm import MyGMM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

k_cluster = 10

X, Y = generate_gaussian_data(k_cluster)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print('LDA with true labels:', acc)
data_test = generate_subset_test_data(X, Y, k_cluster)
X_train, X_test, y_train_sub, y_test = data_test
gmm_model = MyGMM(k_cluster)
gmm_model.fit(X_train, y_train_sub)
y_pred = gmm_model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print('GMM with subset labels:', acc)

