from six.moves import urllib
from sklearn.datasets import fetch_mldata
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ignores future warnings
from scipy.io import loadmat
#mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"

mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

X, y = mnist["data"], mnist["target"]
#the mnist dataset is 70,000 images with 784 features of 28x28 pixels hence 784
import matplotlib
import matplotlib.pyplot as plt 

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

#plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show()
#print(y[36000])

#you should always create a test set and set it saide before inspecting the data closely 
#=> this dataset is already split into training set => first 60,000 is training and last 10,000 is test
X_train, X_test, y_train, y_test = X[:60000], X[:60000], y[:60000], y[:60000]

#shuffle the dataset to ensure cross val folds are similar
import numpy as np 
shuffle_index = np.random.permutation(60000)
X_train, y_train, = X_train[shuffle_index], y_train[shuffle_index]

''' ~~ Training a Binary Classifier ~~ '''
#will detect if the digit is 5 y=1 or not y=0
y_train_5 = (y_train == 5) #True for all 5's false for all other digits
y_test_5 = (y_test == 5)
#now lets pick a clf and train it
#Stochastic Gradient descent is a good start 
#since it can deal with large datasets as SGD deals with training instances independently one at a time
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)#SGD relies on randomness hence stochastic 
sgd_clf.fit(X_train, y_train_5)
#print(sgd_clf.predict([some_digit]))
''' ~~ End of Binary Classifier '''

#Performance measures ~~
#Measuring Accuracy using custom cross-val 
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
	clone_clf = clone(sgd_clf)
	X_train_folds = X_train[train_index]
	y_train_folds = y_train_5[train_index]
	X_test_fold = X_train[test_index]
	y_test_fold = y_train_5[test_index]

	clone_clf.fit(X_train_folds, y_train_folds)
	y_pred = clone_clf.predict(X_test_fold)
	n_correct = sum(y_pred == y_test_fold)
#	print(n_correct / len(y_pred)) #prints 0.9658, 0.962 and 0.96545

#StratifiedKFold performs stratified sampling to produce folds that contain representative ratio of each class
#at each iteration the code creates a clone of the classifier, trains that clone on the training folds and 
#makes predictions on the test fold. Then it counts num of correct predictions and outputs ratio of correct predictions
#stratified sampling  = 'Strata' means 'layer'. A stratified sample is made up of different 'layers' of the population, 
#for example, selecting samples from different age groups. The sample size for each layer is proportional to the size of the 'layer'. 

from sklearn.model_selection import cross_val_score
#cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#[0.9679  0.9667  0.96545]
#Confusion matrix may be better metric
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_train_5, y_train_pred))
#[[53663   916] first row is negative class non-5s    [True Neg, False Neg ]
# [ 1392  4029]] second row positive class            [True Post, False Pos ]
# 53663 correctly classified as negative class i.e. non-5s while 916 were wrongly classified as 5s
# 1392 were wrongly classified as 5s and 4029 correctly classified as 5s

# More accurate is precision
# truepos / truepos + falsepos
# precision is used with recall called true positive rate(TPR)
# truepost / truepos + falseneg

#recall
from sklearn.metrics import precision_score, recall_score 
#print(precision_score(y_train_5, y_train_pred))
#print(recall_score(y_train_5, y_train_pred))

#Better to combine precision and recall into one called F score
#High F score means both precision and recall are high
#F1 = 2 x ( (precision x recall) / (precision + recall) )
from sklearn.metrics import f1_score
#print(f1_score(y_train_5, y_train_pred))
#precision/recall tradeoff increasing one reduces the other
#scikit learn does not let you set the threshold but lets you set the decision scores
#y_scores = sgd_clf.decision_function([some_digit])
#print(y_scores)
# [114847.66767636]
#threshold = 200000 
#y_some_digit_pred = (y_scores > threshold)
#print(y_some_digit_pred)
#False this confirms raising the threhold does raise recall 
#How can you decide which threshold to use? 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#plot precision and recall as functions of the thresholds
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


#plt.figure(figsize=(8, 4))
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.xlim([-700000, 700000])
#plt.show()
y_train_pred_90 = (y_scores > 70000)
#print(precision_score(y_train_5, y_train_pred_90))  0.93
#print(recall_score(y_train_5, y_train_pred_90))  0.638
#That's a 90% precision classifier but high precision is not favorable with low recall

#ROC curve is another common tool
# ~~ the ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) ~~
#The FPR is the ratio of negative instances that are incorrectly classified as postiive. 
#It is equal to 1 - the true negative rate (TNR) which is the ratio of negative instances 
#classified as negative. Hence ROC is sensitivity (recall) vs 1 - specificity
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_pred, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

#plt.figure(figsize=(8, 6))
#plot_roc_curve(fpr, tpr)
#plt.show()

#one way to compare classifier is to measure the area under the curve (AUC)
from sklearn.metrics import roc_auc_score

#print(roc_auc_score(y_train_5, y_scores))   0.94
#You should chose the PR curve whenever the positive class is rare
#Lets train a RandomForest ROC curve
from sklearn.ensemble import RandomForestClassifier
#forest_clf = RandomForestClassifier(random_state=42)
#y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
#the predict proba function is the same as the decision_function()

#y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
#fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
#plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
#plt.legend(loc="lower right", fontsize=16)
#plt.show()

#print(roc_auc_score(y_train_5, y_scores_forest))
#0.993 precision and 82.8% recall

# ~~ Multiclass Classification ~~
#Multiclass/multinomial classifier can distinguish between two or more classes
#Random forest or naive bayes are capable of this SVM's/ linear can only do binary
#one versus all / one vs rest e.g. 1 vs 2,3,4,..,9
#another method is to train One versus One e.g. 1 vs 2, or 1 vs 3, if there are N classes then you need to train N x (N-1)/2

#sgd_clf.fit(X_train, y_train)
#print(sgd_clf.predict([some_digit])) predicts 5
#some_digit_scores = sgd_clf.decision_function([some_digit])
#print(some_digit_scores) 
#[[-153872.03671535 -391193.82756675 -396940.33106115   36538.37618569
 # -396596.77787384   81569.72357841 -571362.11158349 -421121.47881663
 # -521469.65631425 -707999.08386777]]
#The highest scores is the 6th element in that array which is 5 i.e. 0,1,2,3,4,5
#print(np.argmax(some_digit_scores)) 5
#print(sgd_clf.classes_) ~~  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
#when a classifier is trained it stores a list of classes in classes_


from  sklearn.multiclass  import OneVsOneClassifier
#ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
#ovo_clf.fit(X_train, y_train)
#ovo_clf.predict([some_digit])   5
#len(ovo_clf.estimators_)   45

#forest_clf.fit(X_train, y_train)
#forest_clf.predict([some_digit]) # 5 
#Random forest classifier can directly classify intances into multiple classes unlike OvA or OvO
#print(forest_clf.predict_proba([some_digit]))  ~~ [[0.2 0.  0.  0.  0.  0.8 0.  0.  0.  0. ]] its 80% certain its a five
#print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")) get a  84% score
#not a bad score simply scaling the features can increase the score
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
#cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# ~~ Error Analysis ~~
#y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
#conf_mx = confusion_matrix(y_train, y_train_pred)
#print(conf_mx)
#output a confusion matrix
#plt.matshow(conf_mx, cmap=plt.cm.gray)
#plt.show()

#row_sums = conf_mx.sum(axis=1, keepdims=True)
#norm_conf_mx = conf_mx / row_sums
#np.fill_diagonal(norm_conf_mx, 0)
#plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#plt.show()


#cl_a, cl_b = 3, 5
#X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
#X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
#X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
#X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]


# ~~ Multilabel Classification ~~
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]# this code creates a multilabel array with two target labels for each digit image
#the first indicates if an image is large >7
#knn_clf = KNeighborsClassifier()
#knn_clf.fit(X_train, y_multilabel)
#print( knn_clf.predict([some_digit])) False True it gets it right 5 is indeed not large and odd true
#f1 score is a good metric here
#y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
#print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))  to give each label a weight equal to its support i.e.
#the number of instances with that target label set average="weighted"

# ~~ Multioutput classification ~~
# where a label can be multiclass i.e. can have more than two possible values
#start by creating a training set we want to try turn noisy image to clean
'''noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)'''
