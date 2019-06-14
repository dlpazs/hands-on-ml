'''
~~ Ensemble Learning and Random Forests

If you aggregated answer or predictions tends to be better than the wisdom of just one (group wisdom). A group of predictors is called an 
Ensemble. For instance, you can train a group of decision trees on different subset of the training data. To make a prediction you get the 
predictions of all individual trees, then predict the class that gets the most votes. Such an ensemble of decision trees is called a 
Random Forest. 

~~ Voting classifiers ~~

Suppose you've trained a few classifiers each scoring 80% accuracy such as Logistic Regression, SVM, Rnadom forest, K-Nearest.
A simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most 
votes. This majority voting classifier is called a hard voting classifier. 
Suprisingly, this voting classifer often achieves a higher accuracy than the best classifier in the ensemble. 
Even if each classifier is a weak learner, the ensemble can still be a strong learner, provided there are sufficient number of weak
learners and they are sufficiently diverse.
How is this so? The law of large numbers. Suppose you have an ensemble of 1,000 classifiers that are individually correct at 51%
just better than a random guess (heads or tails). If you predict the majority voted class you can get an accuracy of 75%. This is
only true if all the classifiers are perfectly independent, making uncorrelated errors, which isn't correct since they're
trained on the same data. 

'''

from __future__ import division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42, gamma='auto')

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

'''
LogisticRegression 0.864
RandomForestClassifier 0.872
SVC 0.888
VotingClassifier 0.896
Hurrah!! The voting classifier outperforms the individual classifiers.

If all classifiers are able to estimate class probabilities (i.e. they have a predict_proba() method) then you can use Scikit learn to
predict the class with the highest class probability, averaged over all the individual classifiers. THis is called soft voting. It 
often achieves higher performance than hard voting because it gives more weight to highly confident votes. All you need to do is replace
voting="hard" with voting="soft". This is not the case with the SVC class by default, so you need to set it probability hyperparam to True.
This will make the SVC class use cross-validation to estimate class probabilities. 

~~Bagging and Pasting~~
Another approach is to use the same training algorithm for every predictor but to train them on different random subsets of training set.
When sampling is performed with replacement, this method is called bagging (boostrap aggregating). When sampling is performed
without replacement it is called pasting. 
In other words, both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only
bagging allows training instances to be sampled several times for the same predictor. 

clf     clf    clf    clf
subset subset subset subset   (random sampling - with replacement = bootstrap)
    Training Set (whole)

Once all the classifiers/predictors have been trained the ensemble can make a prediction for a new instance by simply aggregating
the predictions of all predictors. The aggregation function is typically the statistical mode (i.e. most frequent prediction). 
Each individual predictor has a higher bias than if it were trained on the original un-bootstrapped training set, but 
aggregation reduces both bias and variance. Generally the net result is that ensemble has a similar bias but a lower variance than
a single predictor trained on the original training set. 

~~Bagging and Pasting in Scikit Learn~~


'''
'''from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))'''
'''
~~Out of bag evaluation~~


'''
