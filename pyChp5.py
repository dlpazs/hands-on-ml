'''
~~ SVMs ~~

Capable of linear and nonlinear classification, regression, and outlier detection. Very good at classification for S/M datasets.

~~ Linear SVM Classification ~~

Classes can be separated (linearly separable). The SVM fits the largest distance between the classes hence called the Large Margin
Classification. Adding more instances wont affect the decision boundary. The instances that are closest to each other are called 
support vectors. SVMs are v v sensitive to feature scales. 

~~ Soft Margin Classification ~~ 

A hard margin classifier imposes that all instances be off the street and on the right hand side.
Two issues with Hard margin => sensitive to outliers. Thus, we need balanced model; one that keeps the street as large as possible
but also limits margin violations(i.e. instances that end up in the middle of the street or the wrong side). 
This is called Soft margin classification. SVM allows you to control this with the C hyperparameter, the smaller the C the wider 
the street => more violations ;). If you SVM is overfitting you can regularise by reducing C.  
'''

'''
import numpy as np 
from sklearn import datasets
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2,3)] #petal length petal width
y = (iris["target"] == 2).astype(np.float64) #Iris Virginica

svm_clf = Pipeline([
	("scaler", StandardScaler()),
	("linear_svc", LinearSVC(C=1, loss="hinge")),
	])

svm_clf.fit(X, y)
'''

#print(svm_clf.predict([[5.5, 1.7]])) [1.]
#You can use SVC class (kernel="linear", C=1) but it is much slower

'''
~~ Nonlinear SVM Classification ~~
Many datasets are not linearly separable so you add poly features. 
Lets try to implement this with the moons datasets.
To implement this using scikitlearn 
'''
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

'''def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])'''

#polynomial_svm_clf.fit(X, y)


'''
~~ Polynomial Kernel ~~

Adding poly features is easy to implement. Low poly cant deal with complex data but too high poly is too slow.
Using the kernel trick SVMs can overcome this. It creates the same result as having many poly features without the combinatorial 
explosion. 
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
	])
#poly_kernel_svm_clf.fit(X,y)
'''
This code trains a SVM with a 3rd degree poly kernel. If model is overfit try reduce degrees of freedom. The hyper parameter
coef0 controls how much the model is influenced by high degree polynomials versus low-degree polynomials. Use grid search to
find the best hyperparams. 

~~ Adding Similarity Features ~~
Another way to tackle nonlinear problems is to add features computed using a similarity function that measures how much each
instance resembles a particular landmark. For example, lets take a 1-d dataset and add 2 landmarks x1 = -2 and x1 = 1. Next, 
lets define the similarity function to be the Guassian Radial Basis Function (RBF) with y = 0.3.
EQUATION: Guassian RBF
phiy(X,L) = exp(-y || X - L||2) [phi subscript y (instance x, legrangian L) = exponential ( -y times magnitude ||x-l||2squared)]
Its a bell shaped function varying from 0 (very far away from the landmark) to 1(at the landmark). Now we can compute new features.
For example, lets look at instance x1 = -1: it is located at a distance of 1 from the first landmark, and 2 from the second 
landmark. Thus, its new features are x2 = exp(-0.1 x 1squared) approaches= 0.74 and x3 = exp(-0.3 x 2squared) approaches= 0.30.
So the equation is the RBF = exponential over minus -y (say y=0.3) times the magnitude of the difference of the
instance x (say x1 = -1) and the landmark (L say L = -2 and 1) squared. 
The RBF transforms the data now so it is linearly separable. How do we select the landmarks? The simplest approach is to create 
a landmark at the location of each and every instance in the dataset. This creates many dismensions and thus increases the chances
that the transformed training set will be linearly separable. The downside is that a training set with m instances and n features
gets transformed into a training set with m instances and m features. 

~~ Gaussian RBF Kernel ~~
The additional features method can be too computationally expensive adding additional features. But the kernel trick overcomes
this by producing the same result of adding features withot actually doing so. 
'''

rbf_kernel_svm_clf = Pipeline([
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
	])
#rbf_kernel_svm_clf.fit(X, y)
'''
This uses gamma(y) and C. Increasing gamma makes the bell-shape curve narrower and as a result instances range of influence
smaller: the decision boundary ends up being more irregular, wiggling around individual instances. Small gamma makes bell-shaped
curve wider, instances have larger range of influence and decision boundary smoother. y acts like a regularistion hyperparameter:
if your model is overfitting you should reduce it, if it is underfitting you should increase it. 
Other kernels  e.g. string kernels. Rule of thumb try linear svc linear kernel first. 

~~ Computational Complexity ~~


~~ SVM Regression ~~
For SVM regression the trick is not to find the largest possible street between two classes whilst limiting margin violations,
SVM tries to fit as many instances as possible on the street while limiting margin violations (off the street). The width of the
street is controlled by hyperparam e (bigger the e bigger the street).
Adding more training instances within the margin doesn't affect the model's prediction; thus the model is said to be e-insensitive. 
You can use scikit learns LinearSVR class to perform this.
'''
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
'''
To tackle nonlin regression tasks you can use kernelized SVM model. 

~~ Under the Hood ~~
Notations: b = bias, w = weights vector
Maths behind SVMS
~~ Decision Function and Predictions
The linear SVM predicts the class of a new isntance x by computing the decision function wT . x + b = w1x1 + ... + wnxn + b.
if y = 0 wT . x + b < 0 , if y = 1 wT . x + >= 0. 

~~ Training Objective ~~
Consider the slope of the descision function it is equal to the norm of the weight vector ||w||. If we divide this slope by 2,
the points where teh decision function is equal to +/- 1 are going to be twice as far away from the decision boundary. In other words,
dividing the slope by 2 will multiply the margin by 2. The smaller the weight vector w the larger the margin. So we want toa assign
||w|| to get a large margin. However, if we also want to avoid any margin violation(hard margin), then we need the decision function
to be greater than 1 for all positive training instances, and lower than -1 for negative training instances. If we define
t(i) = -1 for negatives and t(i) = 1 for positives, if y(i) = 1 then we can express this constraint as t(i)(wT. x(i) + b)>=1 for all
instances. 
EQUATION Hard margin linear SVM classifier objective
minimize w,b   1/2 wT. w subject to t(i)(wT . x(i) + b)>= 1 for i=1,2, ... ,m 
NB: we are minimizing 1/2wT.w, which is equal to 1/2 ||w||2 rather than minimizing ||w|| because it will give the same result 
since the values of w and b that minimize a value also minimize half of its square, but 1/2||w||2 has a nice and simple derivative
(it is just w) while || w || is not differentiable at w = 0. Optimization algorithms work much better on differentiable functions. 
------------------
To get the soft margin we need to introduce the slack variable zeta(i) >= 0 for each instance: zeta(i) measures how much the ith
instance is allowed to violate the margin. We now have two conflicting objectives: making the slack variable as small as possible
to reduce the margin violations, and making 1/2wT . w as small as posible to increase the margin. The C hyperparam allows us to
This gives us the constrained optimization problem in Equation 5-4. 
EQUATION soft margin linear SVM classifier objective. 
minimize w,b,zeta  1/2 wT . w + C SUM m i=1 zeta(i) subject to t(i)(wT . x(i) + b)>= 1 - zeta(i) and zeta(i) >= 0 for i=1,2..,m

~~Quadratic Programming~~ 
The hard margin and soft margin problems are both convex quadratic optimization problems with linear constraints. Such problems
are known as Quadratic Programming (QP) problems. 
EQUATION QP problem
minimize P  1/2pT . H . p + fT . p subject to A . p <= b 

where { 
p is an np - dimensional vector (np = number of parameters),
H is an np x np matrix,
f is an np - dimensional vector,
A is an nc x np matrix (nc = number of constraints),
b is an nc - dimensional vector 
a(i) is the vector containing the elements of the ith row of A and b(i) is the ith element of b. 
}

Note that the expression A . p <= b actually defines nc constraints: pT . a(i) <= b(i) for i - 1,2, ... , nc, where a(i) is the 
vector containing the elements of the ith row of A and b(i) is the ith element of b. 

You can easily verify that if you set the QP parameters in the following way, you get the hard margin linear SVM classifier
objective:
- np = n + 1, where n is the number of features (the +1 is for the bias term),
- nc = m, where m is the number of training instances,
- H is the np x np identity matrix, except with a zero in the top-left cell (to ignore the bias term),
- f = 0 , an np - dimensional vector full of 0s,
- b = 1, an nc - dimensional vector full of 1s,
- a(i) = -t(i) xdotted(i), where xdotted(i) is equal to x(i) with an extra bias feature xdotted0=1

So one way to train a hard margin linear SVM classifier is just to use an off-the-shelf QP solver by passing it the preceding 
parameters. The resulting vector p will contain the bias term b = p0 and the feature weights wi = pi for i = 1,2,...,m. 
Similarly, you can use a QP solver to solve the soft margin problem. 

HOWEVER, to use the kernel trick we are going to look at a different constrained optimization problem. 

~~ The Dual Problem ~~ 

Given a constrained optimisation problem, known as a primal problem, it is possible to express a different but closely related 
problem, called its dual problem. The solution to the dual problem typically gives a lower bound to the solution of the primal
problem, but under some conditions it can even have the same solutions as the primal problem. Luckily, the SVM problem happens
to meet these conditions, so you can choose to solve the primal problem or the dual problem; both will have the same solution.
EQUATION to derive the dual problem from the primal problem using linear SVM objective

minimize alpha 1/2 SUM m i=1 SUM m j=1 alpha(i)alpha(j)t(i)t(j)x(i)T. x(j) - SUM m i=1 alpha(i) subject to alpha(i) >= 0
for i=1,2, .. , m 

Once you find the vector alpha hat that minimizes this equation, you can compute w hat and b hat that minimize the primal problem 
by using EQUATION 5-7 from the dual solution to the primal solution
w hat = SUM m i=1 alpha hat(i)t(i)x(i)
b hat = 1/ns SUM m i=1 (over) alphahat(i)>0 (t(i)-w hatT. x(i))

The dual problem is faster to solve than the primal when the number of training instances is smaller than the number of features.
More importantly, it makes the kernel trick possible, while the primal does not. So what is this kernel trick anyway?

~~ Kernelized SVM ~~

Suppoze you want to apply a 2nd degree polynomial transformation to a 2-d training set, then train a linear SVM classifier
on the transformed training set. 
EQUATION Second degree poly mapping:
phi(X) = phi((x1)) = (  x12       )
			((x2))   ( sqrt2 x1x2 )
					 (  x22       )

notice that the transformed vector is a 3-d instead of 2-d. Now lets look at what happens to a couple of 2-d vectors, a and b,
if we apply this 2nd degree poly mapping and then compute the dot product of the transformed product. 
##KERNEL TRICK FOR SECOND DEGREE POLY MAPPING RESEARCH~~ 

'''