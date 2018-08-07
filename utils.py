import numpy as np
import tensorflow as tf
from numpy.fft import fft, ifft, fftfreq

def RK_timestepper(x,t,f,h,weights,biases,direction='F',method = 'RK4'):
    """
    Explicit Runge-Kutta time integrator.  Assumes no time dependence in f
    """

    if method == 'RK4_38':
        b = [1/8,3/8,3/8,1/8]
        A = [[],[1/3],[-1/3, 1],[1,-1,1]]

    elif method == 'Euler':
        b = [1]
        A = [[]]

    elif method == 'Midpoint':
        b = [0,1]
        A = [[],[1/2]]

    elif method == 'Heun':
        b = [1/2,1/2]
        A = [[],[1]]

    elif method == 'Ralston':
        b = [1/4,3/4]
        A = [[],[2/3]]

    elif method == 'RK3':
        b = [1/6,2/3,1/6]
        A = [[],[1/2],[-1,2]]

    else:
        b = [1/6,1/3,1/3,1/6]
        A = [[],[1/2],[0, 1/2],[0,0,1]]
    
    steps = len(b)

    if direction == 'F':
        K = [f(x, weights, biases)]
        for i in range(1,steps):
            K.append(f(tf.add_n([x]+[h*A[i][j]*K[j] for j in range(i) if A[i][j] != 0]), weights, biases))
    else:
        K = [-f(x, weights, biases)]
        for i in range(1,steps):
            K.append(-f(tf.add_n([x]+[h*A[i][j]*K[j] for j in range(i) if A[i][j] != 0]), weights, biases))

    return tf.add_n([x]+[h*b[j]*K[j] for j in range(steps)])

def RK4_forward(x,t,f,h,weights,biases):
    """
    4th order Runge-Kutta time integrator
    """
    
    return RK_timestepper(x,t,f,h,weights,biases,direction='F',method = 'RK4_classic')

def RK4_backward(x,t,f,h,weights,biases):
    """
    4th order Runge-Kutta time integrator - backwards in time
    """
    
    return RK_timestepper(x,t,f,h,weights,biases,direction='B',method = 'RK4_classic')

def dense_layer(x, W, b, last = False):
    x = tf.matmul(W,x)
    x = tf.add(x,b)
               
    if last: return x
    else: return tf.nn.elu(x)

def simple_net(x, weights, biases):
    
    layers = [x]
    
    for l in range(len(weights)-1):
        layers.append(dense_layer(layers[l], weights[l], biases[l]))

    out = dense_layer(layers[-1], weights[-1], biases[-1], last = True)
    
    return out

def approximate_noise(Y, lam = 10):

	n,m = Y.shape

	D = np.zeros((m,m))
	D[0,:4] = [2,-5,4,-1]
	D[m-1,m-4:] = [-1,4,-5,2]

	for i in range(1,m-1):
	    D[i,i] = -2
	    D[i,i+1] = 1
	    D[i,i-1] = 1
	    
	D = D.dot(D)

	X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[j,:].reshape(m,1)).reshape(1,m) for j in range(n)])

	N_hat = Y-X_smooth

	return N_hat, X_smooth

def get_network_variables(n, n_hidden, size_hidden, N_hat):

    layer_sizes = [n] + [size_hidden for _ in range(n_hidden)] + [n]
    num_layers = len(layer_sizes)

    weights = []
    biases = []

    for j in range(1,num_layers):
        weights.append(tf.get_variable("W"+str(j), [layer_sizes[j],layer_sizes[j-1]], \
                                       initializer = tf.contrib.layers.xavier_initializer(seed = 1)))
        biases.append(tf.get_variable("b"+str(j), [layer_sizes[j],1], initializer = tf.zeros_initializer()))

    N = tf.get_variable("N", initializer = N_hat.astype('float32'))

    return (weights, biases, N)

def create_computational_graph(n, N_hat, net_params, num_dt = 10, method = 'RK4', gamma = 1e-5, beta = 1e-8, weight_decay = 'exp', decay_const = 0.9):

    assert(n == N_hat.shape[0])
    m = N_hat.shape[1]

    ###########################################################################
    #
    # Placeholders for initial condition
    #
    ###########################################################################
    Y_0 = tf.placeholder(tf.float32, [n,None], name = "Y_0")  # noisy measurements of state
    T_0 = tf.placeholder(tf.float32, [1,None], name = "T_0")  # time

    ###########################################################################
    #
    # Placeholders for true forward and backward predictions
    #
    ###########################################################################
    true_forward_Y = []
    true_backward_Y = []

    for j in range(num_dt):
        true_forward_Y.append(tf.placeholder(tf.float32, [n,None], name = "Y"+str(j+1)+"_true"))
        true_backward_Y.append(tf.placeholder(tf.float32, [n,None], name = "Yn"+str(j+1)+"_true"))

    h = tf.placeholder(tf.float32, [1,1], name = "h") # timestep

    ###########################################################################
    #
    #  Forward and backward predictions of true state
    #
    ###########################################################################

    (weights, biases, N) = net_params
    X_0 = tf.subtract(Y_0, tf.slice(N, [0,num_dt],[n,m-2*num_dt]))  # estimate of true state

    pred_forward_X = [RK_timestepper(X_0, T_0, simple_net, h, weights, biases, method = method)]
    pred_backward_X = [RK_timestepper(X_0, T_0, simple_net, h, weights, biases, method = method, direction = 'B')]

    for j in range(1,num_dt):
        pred_forward_X.append(RK_timestepper(pred_forward_X[-1], T_0, simple_net, h, weights, biases, method = method))
        pred_backward_X.append(RK_timestepper(pred_backward_X[-1], T_0, simple_net, h, weights, biases,\
                                            method = method, direction = 'B'))
        
    ###########################################################################
    #
    #  Forward and backward predictions of measured (noisy) state
    #
    ###########################################################################

    pred_forward_Y = [pred_forward_X[j] + tf.slice(N, [0,num_dt+1+j],[n,m-2*num_dt]) for j in range(num_dt)]
    pred_backward_Y = [pred_backward_X[j] + tf.slice(N, [0,num_dt-1-j],[n,m-2*num_dt]) for j in range(num_dt)]

    ###########################################################################
    #
    #  Set up cost function
    #
    ###########################################################################
    
    if weight_decay == 'linear': output_weights = [(1+j)**-1 for j in range(num_dt)] # linearly decreasing importance 
    else: output_weights = [decay_const**j for j in range(num_dt)] # exponentially decreasing importance 

    forward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                          for (w,true,pred) in zip(output_weights,true_forward_Y,pred_forward_Y)])

    backward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                          for (w,true,pred) in zip(output_weights,true_backward_Y,pred_backward_Y)])

    fidelity = tf.add(forward_fidelity, backward_fidelity)

    # Regularizer for NN weights
    weights_regularizer = tf.reduce_mean([tf.nn.l2_loss(W) for W in weights])

    # Regularizer for explicit noise term
    noise_regularizer = tf.nn.l2_loss(N)

    # Weighted sum of individual cost functions
    cost = tf.reduce_sum(fidelity + beta*weights_regularizer + gamma*noise_regularizer)

    # BFGS optimizer via scipy
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={'maxiter': 50000, 
                                                                      'maxfun': 50000,
                                                                      'ftol': 1e-15, 
                                                                      'gtol' : 1e-11,
                                                                      'eps' : 1e-12,
                                                                      'maxls' : 100})

    placeholders = {'Y_0': Y_0,
                    'T_0': T_0,
                    'true_forward_Y': true_forward_Y,
                    'true_backward_Y': true_backward_Y,
                    'h': h}

    return optimizer, placeholders

