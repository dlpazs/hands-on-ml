'''
~~ Up and Running with TensorFlow ~~

Tensorflow is a powerful OOS lib for numerical computation, fine-tuned for large-scale ML. Its basic principle is simple,
First you define in Python a graph of computations to perform and then TensorFlow takes that graph and runs it efficiently
using optimized C++ code.

	 (+)		f(x,y) = x2y + y + 2
   	/   \
   (x)    (+)  operation
   / \   /   \
 (x)	 [y]    [2] 
 /\  variable constant  
[x]
Fig 9.1 A simple computational graph

Most importantly, you can break up the graph into several chunks and run them in parallel across multiple CPUs or GPUs (as show in 
fi 9.2). TF can train a NN with millions of params on a training set of billions of instances with millions of features each.
TF was developed by Google Brain and powers many of Google's large scale services, such as Google Cloud Speech, Google Photos, Google Search.

	f(3,4) = 42
	 /\
	/  \
	 ||
	 (+)		
   	/36 \6
GPU1(x)    (+)  GPU 2
  9/ \4  /4  \2
 (x)	 [y]    [2] 
3/\3  /\  
[x]   ||
/\	  4
||
3
Fig 9.2 Parallel computation on multiple CPUs/GPUs/servers
TF highlights;
-python api called TF.learn, compatible with Scikit-learn. 
-TF-slim to simplify building, training and evaluating NN's.
-Other high level APIs such as Keras/Pretty Tensor
-There is a C++ API to define your own high-performance operations
-It provides several advanced optimization nodes to search for the parameters that minimize the cost function. This is very easy to use
since TF automatically takes care of computing the gradients of the functions you define. This is called automatic differentiating.
-TensorBoard is a great tool for visualising through the computational graph, view learning curves etc.
-TensorFLow graph


~~ Creating your first graph and running it in a session ~~
'''
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

'''x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2'''

'''
The code doesnt actually perform any computation , it just creates a computational graph. Even the variables aren't initialised yet.
To evaluate this graph, you need to open a tensorflow session and use it to initialize the variables and evaluate f. 
A TF session takes care of placing the operations onto devices such as CPUs and GPUs and running them, and it holds all the 
variable values. The following code creates a session, initializes the variables, and evaluates, and f then closes the session
(which frees up resources):
'''
'''sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()'''

#Having to repeat sess.run() all the time is a bit annoying but there's a better way
'''with tf.Session() as sess:
	x.initializer.run()
	y.initializer.run()
	result = f.eval()'''
'''
Inside the with block, the session is set as the default session. Calling x.initializer.run() is equivalent to calling
tf.get_default_session().run(x.initializer).
Instead of manually running the init for every single variable, you can use the global_variables_initializer() function.
This creates a node in the graph to init all vars when it is run:
'''
init = tf.global_variables_initializer() #prepare an init node
'''
with tf.Session() as sess:
	init.run() #actually inits all the vars
	result = f.eval()

print(result)'''
'''
You may prefer to create an InteractiveSession. the difference being that when an interactivesession is created it
automatically sets itself as the default session, so you dont need a with block.
'''
'''sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()'''
'''
A tf program is typically split into two parts: the first part builds a computation graph (the construction phase),
and the second part runs it (the execution phase). 
The construction phase builds a computational graph representing the ML model and the computations required to train it. 
The execution phase generally runs a loop that evaluates a training step repeatedly, gradually improving model params. 

~~ Managing Graphs ~~
Any node you create is auto added to the default graph:
'''
#x1 = tf.Variable(1)
#print(x1.graph is tf.get_default_graph())
'''
You may want to manage multiple independet graps. You can do this by creating a new Graph and temporarily making it default
graph inside a with block:
'''
'''graph = tf.Graph()
with graph.as_default():
	x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())'''
'''
~~ Lifecycle of a Node Value ~~

When you evaluate a node, TF auto determines the set of nodes that it depends on and it evaluates these nodes first. 
For example, consider the following:
'''
'''w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
	print(y.eval()) # 10
	print(z.eval()) # 15'''

'''
First, this code defines a very simple graph. Then it starts a session and runs the graph to evaluate y: TensorFlow automatically
detects that y depends on x, which depends on w, so it first evaluates w, then x, then y, and returns the val of y. Finally,
the code runs the graph to evaluate z. Once again, TensorFlow detects that it must first evaluate w and x. It is important to note
that it will not reuse the result of the previous evaluation of w and x. In short, the preceding code evaluates w and x twice. 

All node values are dropped between graph runs, except variable values, which are maintained by the session across graph runs.
A variable starts its life when its initializer is run, and it ends when the session is closed. 

If you want to evaluate y and z efficiently, without evaluating w and x twice as in the previous code, you must ask TF to eval
both y and z in just one graph run:
'''
'''with tf.Session() as sess:
	y_val, z_val = sess.run([y, z])
	print(y_val)
	print(z_val)'''

'''
In single process TF, multiple sessions do not share any state, even if they reuse the same graph(each session would have its own
copy of every variable). Variable state is stored on the servers, not sessions, so multiple sessions can share the same vars.

~~ Linear Regression with TensorFlow ~~

TF operations called ops, can take any number of inputs and produce any number of outputs. For example, the addition and multiplication
ops each take two inputs and produce one output. Constants and variables take no input theyre called source ops. 
The inputs and outputs are multidimensional arrays, called tensors. Just like Numpy arrays, tensors have a type and shape.
In fact, in the Python API tensors are simply represented by Numpy ndarrays. They typically contain floats, but you can also use them
to carry strings (arbitrary byte arrays).

So far we have only dealt with tensors contained a single scalar value, but you can perform computations on arrays of any shape. 
The following code manipulates 2D arrays to perform lin reg. Starts by fetching the dataset; then it adds an extra bias input 
feature (x0 = 1) to all training instances(it does so using Numpy so it runs immediately); then it creates two TF constant nodes,
X and y, to hold this data and the targets, and it uses matrix operations to define theta. These functions transpose(), matmul(),
and  matrix_inverse() - do not perform any computations immediately, instead they create nodes in the graph that will perform
them when the graph is run. You may recognise that the definition of theta corresponds to the Normal equation. Finally, the code
creates a session and uses it to evaluate theta. 
'''
import numpy as np 
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
	theta_value = theta.eval()
'''
~~ Implementing Gradient Descent ~~


'''