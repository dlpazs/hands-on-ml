'''
~~ Decision Trees ~~

~~ Training and Visualizing a Decision Tree ~~

'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] #petal length and width
y = iris.target 

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

'''
~~ Making Predictions ~~

How does it make predictions? It starts at the root node, asks if the petal length is smaller than 2.45cm. If it is then
move down the root's left child node. 
One of the benefits of Decision trees is they require very little data preparation in terms of feature scaling, or centering.
A node's sample attribute counts how many training instances it applies to.
A node's value attribute tells you how many training instances each class this node applies to. 
A node's gini attribute measures its impurity: a node is pure (gini=0) if all training instances it applies to belong to the same class.

EQUATION Gini Impurity
Gi = 1 - SUM n k=1 Pi,k2
Pi,k is the ratio of class k instances among the training instances in the ith node

~~ Estimating Class Probabilities ~~

A decision tree can also estimate the probability that an instance belongs to a particular class k: first it traverses the tree to find
the leaf node for this instance, and then it returns the ratio of training instances of class k in this node. 

'''

#print(tree_clf.predict_proba([[5,1.5]]))[[0.         0.90740741 0.09259259]]

'''
~~ The CART training algorithm ~~

The classification and regression tree algorithm to train decision trees(also called growing trees). 
The algorithm tries first splits the training set in two subsets using a single feature k and a threshold tk(e.g. petal length<=2.45cm).
How does it choose k and tk? it searches for the pair (k, tk) that produces the purest subsets (weighted by their size). The cost
function it tries to minimize is as follows;
EQUATION CART cost function for classification
J(k, tk) = mleft/m * Gleft + mright/m * Gright
where{ Gleft/right measures the impurity of the left/right subset,
mleft/right is the number of instances in the left/right subset}

Once it has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets and so on, recursively.
It stops recursing when it reaches the maximum depth (given by the max_depth hyperparameter), or if it cant find a split that will reduce
impurity. 
The CART algorithm is a greedy algorithm: this means that it greedily searches for the optimal split at the top level, then repeats the
process at each level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. It does
not guarantee an optimal solution. Finding an optimal tree is known as an NP-Complete problem: it requires O(exp(m)) time, making the
problem hard to deal with even small training sets. 

~~ Computational Complexity ~~

Making predictions requires traversing the tree from root to leaf. Decision trees are generally balanced, so traversing requires
going through roughly O(log2(m)) nodes. However, the training alg compares all features on all samples at each node.
THis results in training complexity of O(n x m log(m)).

~~ Gini Impurity or Entropy ~~

By default Gini is used but you can select the entropy impurity measure instead by setting the criterion hyperparameter to entropy.
A set's entropy is 0 when it contains instances of only one class. 
EQUATION Entropy

Hi = - n SUM k=1 Pi,k not equal 0 Pi,k log(Pi,k) 

Gini or cross entropy? both the same but gini slightly faster.

~~ Regularization Hyperparamters ~~

Decision trees make very few assumptions about the training data and are prone to overfit. Such a model is called nonparametric
model, called because the number of parameters is not determined prior to training, so the model structure is free to strick closely to
the data. In contrast, a parametric model such as linear model has predetermined number of parameters, so its degree of freedom
is limited. 

To avoid overfit you need to restrict the trees degrees of freedom. Controlled by max_depth hyper param. Others;
min_samples_split(the min number of samples a node must have before it can be split), min_samples_leaf(number of samples a leaf must have),
min_weight_fraction_leaf, max_leaf_nodes.. 
Increasing min_* or decreasing max_* will regularise the model. 
Other algorithms prune (delete) uneccessary nodes if the purity improvement is not staistically significant.

~~ Regression ~~

Decision trees are also capable of regression. 

'''

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

'''
Instead of predicting a class it predicts a value. The value is simply the average target value of the training instances associated to 
the leaf node. 

~~ Instability ~~

Limitations: Decision trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive
to training set rotation. 
Dec Trees are sensitive to small variations in the training data. 

'''
