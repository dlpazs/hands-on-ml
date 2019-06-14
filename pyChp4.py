# ~~ Chapter 4 ~~ Training Models
'''
~~ Linear Regression ~~
yhat = theta0 + theta1x1 + theta2x2 + ... + thetaNxN
theta 0, 1 are the models params
- a linear model makes a prediction of the weighted sum of the input features plus a constant term called the bias (intercept term)
- yhat = predicted value, n is the number of features, xi is the ith feature value, thetaj is the jth model parameter including the 
bias term theta0 and feature weights theta1, theta2, ... thetaN
**the vectorized linear equation model
 - yhat = htheta(x) = thetaTranspose dot product x

theta = models params vector, thetaTranspose = theta transpose into row vector instead of column, 
x is the instance's feature vector x1,..xn, thetaT . x is the dot product of thetaT and x, htheta is hyptothesis

Now thats the model how do we train it? Well we first need to measure how well the model fits the training data. We can use the RMSE, or MSE
MSE cost function for linear regression:
MSE(X, htheta) = 1/m sum m i=1 (thetaTranspose dot product x(i) - y(i))2

~~ Normal Equation ~~

To find the value of theta that minimizes the cost function, there is a mathematical equation that gives the result directly => normal equation
Thetahat = (XT . X)-1 . XT . y
y is the vector of target values containing y(1) to y(m)
thetahat is the value of theta that minimizes the cost function
'''
#lets generate some linear looking data
'''import numpy as np

X = 2 * np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
#now lets compute thetahat using normal equation. We will use the inv() function of np to compute the inverse of a matrix 
#and dot() for mat multiplication

X_b = np.c_[np.ones((100, 1)), X] #add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#print(theta_best)
#[[4.03459037]
# [3.08720768]]
#Now you can make predictinos using thetahat
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new] #add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
#print(y_predict)
#[[3.97061589]
# [9.8808136 ]]
#Lets plot the predictions
import matplotlib.pyplot as plt 
'''
'''plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()'''

#Equivalent scikit learn code
'''from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
#print(lin_reg.intercept_, lin_reg.coef_) #[4.10621833] [[2.90053325]]
#print(lin_reg.predict(X_new)) 
#[[4.10621833]
# [9.90728484]]

'''
'''~~ Computational Complexity ~~

The normal equation computes the inverse of XT . X, is a nxn matrix and the computational complexity of inverting such a matrix is 
O(n2.4) to O(n3)

~~ Gradient Descent ~~
The basic idea behind gradient descent is to tweak parameters iteratively to minimize the cost function

'''
# Ignore useless warnings 
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

#Plots the line of best fit
#plt.plot(X_new, y_predict, "r-")
#plt.plot(X, y, "b.")
#plt.axis([0, 2, 0, 15])
#plt.show()
#can do the above with below
#line reg equat yhat = theta0 (bias) + theta1x1 + .. thetanxn
#vectorised linreg is yhat = htheta(x) = theta transpose the dot product of x
#How well did the model fit the data? How to set the params so that the model best fits the training set
#We use the Mean Square Error (MSE) 
#MSE cost function: MSE(X, htheta) = 1/m sum m i=1 (theta transpose dot product x(i) -y(i))2
#To find the value of theta that mins the cost function we can use the Normal equation: theta hat = (X transpose dot product X)-1 dot product X transpose dot product y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#print(lin_reg.intercept_, lin_reg.coef_)  ~~ [4.11253303] [[2.76190641]]
#print(lin_reg.predict(X_new))
#[[ 4.18677157]
# [10.08491783]]

#Some level of computational complexity with using the normal equation
#The normal equation gets very slow when the numb of features grows large
# ~~ Gradient Descent ~~
#Tweaking parameters iteratively to minimize the cost function
#Direction of steepest descent
#Error function is the line function and it measures the local gradient of that error function with regards
#to the parameter vector theta and goes in the direction of descending gradient until the gradient is 0
#You start by filling theta with random values i.e. random initialisation and take gradual steps until convergence
#The learning rate hyperparameter determines the size of the steps
#THe MSE is a convex cost function i.e. one global minimum
#Use feature scaling i.e. StandardScaler for GD

# ~~ Batch gradient descent ~~
#to implement GD you need to compute the gradient of the cost function with regards to each model param thetaj
# in other words you need to calculate how much the cost functin will change in response to a small change in thetaj
# THis is called the partial derivative i.e. "what is the slope of the mountain under my feet if i face east? What about West/North/etc"
#Equation: d/d thetaj MSE(theta) = 2/m sum m i=1(theta Transpose dot product x(i) - y(i) )x(i)j
#We compure the partial derivaties all in one go for each model parameter  ~~ Hence batch GD
#Once you have the gradient vector that points uphill go in the opposite direction downhill
#This means subtracting Vthetasubscript MSE(theta) from theta 
#Multiply the gradient vector by n (i.e learning rate) to determine the size of the downhill step

eta = 0.1 #learning rate
n_iterations = 1000 
m = 100 #training examples
theta = np.random.randn(2,1) #random initialised theta

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) #gradients is the partial derivative the direction of steepest descent
    theta = theta - eta * gradients #then theta that mins cost function is theta minus learning rate times the gradient
#eta * gradients determines the size of downhill step
#print(theta)
#Use GridSearch to find the learning rate
#~~ Stochastic Gradient Descent ~~
# Batch can be too time consuming
# Stochastic GD picks a random instance in the training set at every step and computes the gradients based on that one instance
# Good for large training sets only one instance needs to be in memory at each iteration
# due to its random (stochastic) nature it bounces around and once it gets close to the minimum it bounces out again
#so once it stops the final params are good but not optimal
#One solution is to gradually reduce the learning rate step size. This is called "simulated annealing"
#the function that determines the learning rate at each iteration is called the learning schedule
n_epochs = 50
t0, t1 = 5, 50 #learning schedule hyperparameters
def learning_schedule(t):
	return t0 / (t + t1)

theta = np.random.randn(2,1)#random init

for epoch in range(n_epochs):
	for i in range(m):
		random_index = np.random.randint(m)
		xi = X_b[random_index:random_index+1]
		yi = y[random_index:random_index+1]
		gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
		eta = learning_schedule(epoch * m + i)
		theta = theta - eta * gradients

#print(theta)
#To use SGD with lin reg use SGDRegressor 
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42) #runs 50 epochs, learning rate 0.1, default learning schedule and no regularisation i.e. penalty=none
sgd_reg.fit(X, y.ravel())
#print(sgd_reg.intercept_, sgd_reg.coef_)

#~~ Mini Batch GD ~~
'''
Computes the gradients on small random sets of instances - so not just on one (SGD) or the whole thing (BGD).
less erratic than SGD and closer to the minimum. BUT, harder to escape from local minima when problems suffer from local minima.
Algorithm 	large (m) 	out-of-core support 	large (n)	hypereparams	scaling required	scikit-learn
~Normal equation Fast	No 						slow        	0 				no 					LinearRegression
~ Batch GD  	slow     	no 						fast        2               Yes                   n/a
SGD              fast        yes                   fast        >=2                yes                  SGDRegressor
Mini-batch GD     fast          yes                 fast        >=2               yes                   SGDRegressor
'''

''' ~~ Polynomial Regression ~~

What is data is nonlinear ~ well you can fit linear regression to non-linear data by:
adding the powers of each feature as new features => polynomial regression.
Lets start by creating some non-linear data
'''
import numpy as np
import numpy.random as rnd

np.random.seed(42)
m = 100 
X = 6 * np.random.randn(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
'''
Lets use scikit learns PolynomialFeatures class to transform our training data
 Polynomial with degree=3 adds not only features a2, a3, b2, b3 but also the combination ab, a2b, ab2
 PolynomialFeatures(degree=d) transforms an array containing n features into an array containing n! where
 n!= n factorial of n equal to 1 x 2 x 3 x ... x n => beware of the combinatorial feature explosion
'''
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
#print(X[0]) [-0.7781204]
#print(X_poly[0])

#Plotting the linear regression agains the poly
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

#plt.plot(X, y, "b.")
#plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
#plt.xlabel("$x_1$", fontsize=18)
#plt.ylabel("$y$", rotation=0, fontsize=18)
#plt.legend(loc="upper left", fontsize=14)
#plt.axis([-3, 3, 0, 10])
#plt.show()

'''
Learning Curves
 High degree of poly will fit much better than lin reg 
 lin reg underfit ~ poly overfit => quadratic best fit
 How can you tell if your model is underfitting or overfitting?
We used cross-validation to estimate a model's generalisation performance. If a model performs well on the training data but 
generalises poorly according to cross-valdiation then your model is overfitting, if it performs poorly on both then it is underfit.
Another way to measure this is learning curves. These are plots of the models performance on the training set and validation set as
a function of the training set size (or the training iteration). 
'''
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	train_errors, val_erros = [], []
	for m in range(1, len(X_train)):
		model.fit(X_train[:m], y_train[:m])
		y_train_predict = model.predict(X_train[:m])
		y_val_predict = model.predict(X_val)
		train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
		val_erros.append(mean_squared_error(y_val_predict, y_val))
	#plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
	#plt.plot(np.sqrt(val_erros), "b-", linewidth=3, label="val")

lin_reg = LinearRegression()
#plot_learning_curves(lin_reg, X, y)
#plt.show()
#The model starts to plateau at which point new instances dont improve performance 
#Now lets see the learning curves for a 10th degree poly
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

#plot_learning_curves(polynomial_regression, X, y)
#plt.axis([0, 80, 0, 3])           
#plt.show()                        
#The error here is much lower and there is a gap between the curves. This means the model performs better on training data
# than on the validation data, which => overfitting. One way to fix this is more data
'''
~~ The Bias / Variance Trade off ~~

Bias: High bias means the model is underfit => wrong assumptions, e.g. the data is linear when it is actually quadratic
Variance: Excessive sensitivity to small varitions in the training data. A model with many degrees of freedom
like a higher order poly => overfit
Irreducible Error: relates to noisiness of data. One way to reduce this is to clean up the data i.e. remove outliers
NB: increasing model complexity normally increases variance reduces bias and vice versa

~~ Regularized Linear Models ~~
One way to reduce overfit is regularize the model i.e. to constrain it: the fewer the degrees of freedom it has the harder
it will be to overfit the data. E.g. an easy way to regularize a polynomial model is to reduce the number of poly degrees.
For a linear model, regularisation is normally done by constraining the weights.

~~ Ridge Regression ~~
is a regularised version of linera regression: a regularisation term alphaSUM n i=1 theta2i is added to the cost function.
This forces the learning algorithm to fit the data but also keep the model weights as small as possible. 
Once trained you evaluate by using the unregularised performance measure. 
The hyper parameter alpha controls how much you want to regularise the model. if alpha = 0 then Ridge Regression is just
linear regression. if alpha is very large than all weights end up very close to zero.
EQUATION:
J(theta) = MSE(theta) + alpha 1/2 SUM n i=1 theta2j
You dont regularise the bias term theta0 but start at i=1 not 0. 
If we define w as the vector of feature weights (theta1, thetaN) then the regularisation term is simply equal to 
1/2(||w||2)2 where ||.||2 represents the l2 norm of the weight vector. 
l2 norm = ||.||2 = sqrt B20 + B21 
l1 norm = ||B||1 = |B0|+ |B1|   see Vector Norms l2 Norm => a circle for l2 and diamond for l1
increasing alpha leads to flatter line 
NB: Euclidean distance is the shortest path between two points.
Closed form solution to Ridge Regression:
thetahat = (XT . X + alphaA)-1 . XT . y
Heres how in scikit learn
'''
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
#print(ridge_reg.predict([[1.5]])) [[3.67075511]]
#using SGD
sgd_reg = SGDRegressor( penalty="l2")
sgd_reg.fit(X, y.ravel())
#print(sgd_reg.predict([[1.5]])) [-0.0105826]
'''
~~ Lasso Regression ~~
Least Absolute Shrinkage and Selection Operator Regression => LASSO
it uses l1 norm instead of l2
EQUATION:
J(theta) = MSE(theta) + alpha SUM n i=1|thetai|
Lasso eliminates weights of the least important features i.e. set them to zero
'''
from sklearn.linear_model import Lasso 
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
#print(lasso_reg.predict([[1.5]]))[3.68274416]
'''
~~ Elastic Net ~~
is the middle ground between ridge and lasso. It is a mix of the two, and you can control the mix ratio r. 
Where r = 0 => ridge and r = 1 => lasso
J(theta) = MSE(theta) + ralpha SUM n i=1|thetai|+ 1-r/2 alpha SUM n i=1 theta 2 j
It is always preferable to have a little bit of regularisation. If you suspect only a few features are useful use
lasso/elastic. 
'''
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
#print(elastic_net.predict([[1.5]]))	[3.70117552]
'''
~~ Early Stopping ~~
a very different way to stop an iterative learning algorithm like gradient descent is to stop training as soon as the validation
error reaches a minimum => early stopping. NB: with mini batch and stochastic the curves are not smooth so it may be hard to
tell if you've reached a minimum. One solution is to stop only after the validation error has been above the minimum for some time
then roll back. 
'''
from sklearn.base import clone 
#prepare the data
'''poly_scaler = Pipeline([
	("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
	("std_scaler", StandardScaler()) ])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None 
best_model = None 
for epoch in range(1000):
	sgd_reg.fit(X_train_poly_scaled, y_train) #continues where it left off
	y_val_predict = sgd_reg.predict(X_val_poly_scaled)
	val_error = mean_squared_error(y_val_predict, y_val)
	if val_error < minimum_val_error:
		minimum_val_error = val_error
		best_epoch = epoch 
		best_model = clone(sgd_reg)
#when warm_start=True it continues training where it left off
'''
'''
~~ Logistic Regression ~~
Logistic reg is used to estimate the probability that an instance (x) belongs to a class (y). If estimated probability is 
greater than 0.5 then positive y=1. 

Estimating Probabilities
Just like lin reg log reg computes the weighted sum of the input features plus the bias term BUT instead of outputting the result
it outputs the logistic of this result. 
Vectorised EQUATION: 
phat = htheta(x) = sigmoid(thetaT . x)
Once Log reg has estimated the probability that phat = htheta(x) that an instance x belongs to a positive class in can make the 
prediction yhat easily. NB: notice that sigmoid(t) < 0.5 when t < 0 and sigmoid(t) >= 0.5 when t >= 0, so log reg predicts 
1 if thetaT . x is positive and 0 if it is negative. 

Training and Cost Function
How does it train? The objective of training is to set parameter vector theta so that the model estimates high probabilities for 
positive instances and low for negative instances. 
Cost function:
c(theta) = { -log(phat) if y=1 or -log(1 - phat) if y-0 }
This cost function makes sense because log(t) grows very large when t approaches 0, so the cost will be large if the model estimates
a probability close to 0 for a positive instance, and it will also be very large if the model estimates a probability close to 1
for a negative instance. On the other, log(t) is close to 0 when t is close 1, so the cost will be close to 0 if the estimated 
probability is close to 0 for a negative instance or close to 1 for a positive instance. 
The cost function over the whole training set is simply the average cost over all training instances. It can be written as a single
expression called the log loss;
J(theta) = -1/m SUM m i=1 [y(i)log(phat(i)) + (1 - y(i))log(1-phat(i))]
The cost function is convex so the cost function is guaranteed to find the global minimum eventually. 
The partial derivatives of the cost function with regards to the jth model parameter thetaj;
d/dthetaj J(theta)=1/m SUM m i=1 (sigmoid(thetaT.x(i) - y(i))xj(i)
For each instance it computes the prediction error and multiplies it by the jth feature value, and then computes the average over 
all training instances. 
'''

'''
~~ Decision Boundaries ~~
lets use iris for log reg
'''

'''
~~ Iris Data Set ~~
'''
from sklearn import datasets
iris = datasets.load_iris()
#print(list(iris.keys()))['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
#X = iris["data"][:, 3:]#petal width
#y = (iris["target"] == 2).astype(np.int) # 1 if iris virginica else 0 CAN CHANGE
from sklearn.linear_model import LogisticRegression

#log_reg = LogisticRegression()
#log_reg.fit(X, y)
#X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
#y_proba = log_reg.predict_proba(X_new)
#plt.plot(X_new, y_proba[:,1], "g-", label="Iris-Virginica")
#plt.plot(X_new, y_proba[:,0], "r--", label="Not Iris-Virginica")
#plt.show()
#The decision boundary is around 1.6cm for petal width
#print(log_reg.predict([[1.7],[1.5]])) [1, 0] [+, -]
'''
Log reg models can be regularised and scikit learns has an l2 penalty by default.

~~ Softmax regression ~~
the log reg model can be generalised to support multiple classes without having to train and combine multiple binary classifiers.
This is called Softmax reg or Multinomial Log reg. 
When given an instance x, the Softmax reg model first computes the a score sk(x) for each class k, then estimates the probability
of each class by applying the softmax function (also called normalised exponential) to the scores. 
EQUATION:
Sk(x) = (theta(k))T . x
NB: Each class has its own dedicated parameter vector theta(k). All these vectors are typically stored as rows in a parameter matrix (-).
Once you have computed the score of every class for instance x, you can estiamte the probability phatk that the instance belongs to 
class k by running the scores through the softmax function: it computes the exponential of every score then normalizes them (dividing 
by the sum of all exponentials). 
EQUATION:
phatk = sigmoid(s(x))k = exp(sk(x))/SUM kj=1 exp(sj(x))
K = num of classes
s(x) is a vector containing the scores of each class for the instance x
sigmoid(s(x))k is the estiamted probability that the instance x belongs to class k given the scores of each class for that instance.
EQUATION for softmax prediction (predicts the class with the highest estimated probability=> class with highest score):
yhat = argmaxk sigmoid(s(x))k = argmaxk((theta(k))T . x)
The argmax operator returns the value of a variable that maximizes a function. It returns the value of k that maximises the estimated
probability that, sigmoin(s(x))k , instance x belongs to class k given scores of each class for that instance.
Lets use cross entropy cost function to penalise the model when it estimates low probability for a target class. Cross entropy
is used to measure how well a set of estimates class probabilities match the target classes. 
EQUATION Cross entropy cost function:
J((-)) = - 1/m SUM m i=1 SUM k k=1 y(i)k log(phat(i)k)
y(i)k is equal to 1 if the target class for the ith instance is k; otherwise it is equal to 0

Cross Entropy
originated from information theory. Suppose you want to transmit info about weather everday. 8 options (sunny, rainy etc.), you 
could encode each using 3 bits since 2cubed = 8. However, if you think it will be sunny more often than not it is more efficient to
code sunny on just one bit (0) and the other seven options on 4 bits. Cross entropy measures the number of bits you actually send 
per option. If your assumption is correct it will be equal to the entropy of the weather itself (i.e. its intrinsic unpredictability).
But if your wrong, cross entropy will be greater by an amount called Kullback-leibler divergence. 
The cross entropy between two probability distributions p and q is defined as H(p,q) = - SUMxP(x)log q(x) 
EQUATION gradient vector of CE for class k:
Vtheta(k)J((-)) = 1/m SUM m i=1 (phat(i)k - y(i)k)x(i)  
'''
#Log reg uses one-versus-all by default
#lets use Softmax for multinomial reg

X = iris["data"][:, (2,3)]
y = iris["target"] 
#lbfgs is the solver that support softmax reg
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
#print(softmax_reg.predict([[5,2]]))#[2]
#print(softmax_reg.predict_proba([[5,2]])) [[6.38014896e-07 5.74929995e-02 9.42506362e-01]]
