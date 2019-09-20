
import numpy as np

def stock_sim_path(S, alpha, delta, sigma, T, N, n):
    """Simulates geometric Brownian motion."""
    import random
    random.seed(12345)
    h = T/n
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))

figdir = '/Users/mhosseini/Documents/projects/book/figs'    #figures directory, used by LaTex
S=49.; K=50.; r=0.05; dt=0.3846; v=0.2 
t = 1 #years of training data
days = int(252*t)
stock_path = stock_sim_path(S, .05, 0, v, dt, 1, days) #simulate stock path
stock_path_test = stock_sim_path(S, .05, 0, v, dt, 1, days) #simulate stock path for cross-validation
#plot stock paths
stock_path_mean = [np.mean(stock_path)]*len(stock_path)
stock_path_test_mean = [np.mean(stock_path_test)]*len(stock_path_test)

from matplotlib import pyplot as plt
fig,ax = plt.subplots()
# Plot the data
plot1 = ax.plot(stock_path,'k')
plot2 = ax.plot(stock_path_test,'k--')
plot3 = ax.plot(stock_path_mean,'r--')
plt.title(r'$S_{t_n} = S_0 e^{(\alpha-\frac{1}{2}\sigma^2)(t_n - t_0)} + \sigma \sum_{i=1}^n Z_i \sqrt{t_i - t_{i-1}} $')
plt.ylabel(r'GBM Stock Prices ($)')
plt.xlabel('Trading Days')
plt.legend(['training sample', 'testing sample', 'Average of Training'], loc='lower right',frameon=False)
plt.gca().spines['right'].set_color('none')     #remove top borderline
plt.gca().spines['top'].set_color('none')       #remove right borderline
#plt.savefig(figdir+'/GBM')  #uncomment to save a new plot 
plt.show()

'''
At this point we have generated both training and test stock data.
 These are now used to generate BS theoretical option prices
'''
def get_batch(stock_path,n, moneyness_range = (.5,2)):
    """Constructs theoretical options based on the time series stock_path"""
    picks = np.random.randint(0, len(stock_path)-1, n)
    t = np.random.randint(1, 150, (n,1))
    S = stock_path[picks]
    S_ = stock_path[picks+1]
    K = np.random.uniform(*moneyness_range, (n,1))*S
    X = np.hstack([S, K, t/252])
    X_ = np.hstack([S_, (t-1)/252])
    return X, X_

batch_size = 1000   #number of theoretical options in each batch
X_test, X_test_ = get_batch(stock_path_test, batch_size) #get test-set

n_epochs = 50       #number of training epochs
n_batches = 100     #number of batches per epoch

#neural net for the option's synthetic data
import tensorflow as tf
activation = tf.tanh #activation function
hidden_layer = [50,50,50,50,50] #number neurons in each hidden layer
n_outputs = 1
learning_rate = .001
tf.reset_default_graph()

#The training requires the value of S and T at both times t and (t+h),
#   the latter is denoted by a an underscore "_" at the end
with tf.name_scope('inputs_processing'):
    X_input = tf.placeholder(tf.float32, shape = (None, 3), name = 'X_input')   #S, K, t
    X_input_ = tf.placeholder(tf.float32, shape = (None, 2), name = 'X_input_') #S_, t_
    r = tf.fill([tf.shape(X_input)[0],1], 0., name = 'r')                       #r

#input matrix for ANN
S = tf.slice(X_input, (0,0), (-1,1))
K = tf.slice(X_input, (0,1), (-1,1))
t = tf.slice(X_input, (0,2), (-1,1))
X = tf.concat([S/(K*tf.exp(-r*t)), t], 1)
#input matrix for ANN_
S_ = tf.slice(X_input_, (0,0), (-1,1))
t_ = tf.slice(X_input_, (0,1), (-1,1))
X_ = tf.concat([S_/(K*tf.exp(-r*t_)), t_], 1)

with tf.name_scope('ann'):
    #defines the nerual network architecture, inputs are S/K and T, output is C/K
    def ann(x, hidden_layer, n_outputs, activation, reuse = False):
        Z = tf.layers.dense(x, hidden_layer[0], 
                            activation = activation, name =  'hidden1', reuse = reuse)
        for i in range(1, len(hidden_layer)):
            Z = tf.layers.dense(Z, hidden_layer[i], 
                                activation = activation, name = 'hidden' + str(i+1), reuse = reuse)
        return tf.layers.dense(Z, n_outputs, name = 'out', reuse = reuse)

    out = ann(X, hidden_layer, n_outputs, activation) #out is ANN estimate of C/K
    out = tf.where(tf.greater(t, 1e-3), out, tf.maximum(S/K - 1, 0)) 
        #if T<0.001 (basically if T==0), then max(S/K-1,0) is returned instead of ANN estimate
    out = K*out # multiply (C/K) by K to obtain C

    #derivatives of option price is computed
    delta = tf.gradients(out, S)[0]
    theta = tf.gradients(out, t)[0]
    gamma = tf.gradients(delta, S)[0]

    #same as above, but for option price at (t+h)
    out_ = ann(X_, hidden_layer, n_outputs, activation, reuse = True)
    out_ = K*tf.where(tf.greater(t_, 1e-3), out_, tf.maximum(S_/K - 1, 0))

with tf.name_scope('loss'):
    #this is the loss (objective) function with delta only:
    hedging_mse = tf.losses.mean_squared_error(labels = delta*(S_-S),
                                               predictions = (out_-out)) 
with tf.name_scope('loss'):
    #this is the loss (objective) function with all the Greeks:
    bs_hedging_mse = tf.losses.mean_squared_error(labels = delta*(S_-S)+0.5*gamma*(S_-S)*(S_-S)+theta*t/days,
                                               predictions = (out_-out)) 
with tf.name_scope('training'):
    #using ADAM optimization
    optimizer = tf.train.AdamOptimizer(learning_rate) 
    training_mse = optimizer.minimize(hedging_mse)

with tf.name_scope('init_and_saver'):
    #here: create a node in the netowrk to initialize all the tensorflow variables
    #later: when the session is active, perform the actual initialization.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


with tf.Session() as sess: #start tensorflow session
    init.run() #initialize variables
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_train, X_train_ = get_batch(stock_path, batch_size) #get batch of theoretical options
            sess.run([training_mse], feed_dict = {X_input: X_train, X_input_: X_train_}) #training operation

        epoch_loss = bs_hedging_mse.eval({X_input: X_test, X_input_: X_test_})
        
        print( 'Epoch: ' , epoch, 'Loss: ' , epoch_loss,  'BS Loss:' ,
              bs_hedging_mse.eval({X_input: X_test, X_input_: X_test_}))    
        
        #res = {X_input: X_train} 
        #print('res:',res)
        
    save_path = saver.save(sess, figdir +  '/ann_save.ckpt' ) #save model parameters













def BSMPricer_DL(S,K,r,t,v):
    ''' 
        Author: Matthew Hosseini
        Date:   July 31, 2019
        Guide:
            
        Using Deep-Learning to: 
            
            1) price vanilla call options, and
            2) compute sensitivities. 
    
        The price and related sensitivities are computed by: 
            
            1) initializing the required parameters:
                S=49.; K=50.; r=0.05; dt=0.3846; v=0.2 
            2) invoking the function:
                c = BSMPricer_DL(S,K,r,dt,v); print([c])
            3) and observing the output:
                   [[2.4004612, 
                     [0.5216017, 4.3053904, 12.105244], 
                     [0.06554538, 0.19676477, 0.1391429]]]
            
            where: 
                    from Tensor 0: [2.4004612] ... so call = $2.40 
                    from Tensor 1: [0.5216017, 4.3053904, 12.105244] 
                        ... so \delta = 0.5216 \theta = -4.3054  \vega = 12.1052 
                    from Tensor 2: [0.06554538, 0.19676477, 0.1391429]
                        ... so \gamma = 0.0499 charm = 0.1968   vomma = 01391
    '''
    import tensorflow as tf
    #create the input variables
    S_ = tf.placeholder(tf.float32)
    K_ = tf.placeholder(tf.float32)
    r_ = tf.placeholder(tf.float32)
    t_ = tf.placeholder(tf.float32)
    v_ = tf.placeholder(tf.float32)
    N_ = tf.distributions.Normal(0.,1.).cdf
    #create BSM's price formula
    d1_ = (tf.log(S_ / K_) + (r_+v_**2/2)*t_) / (v_*tf.sqrt(t_))
    d2_ = d1_ - v_*tf.sqrt(t_)
    oprice = (S_*N_(d1_) - K_*tf.exp(-r_*t_)*N_(d2_))  
    #create a matrix where the gradients (computed greeks) are kept
    grad = [oprice]    
    #1st order partials are: delta,theta,vega 
    greeks_1storder_partials = tf.gradients(oprice, [S_,t_,v_]) 
        #tf.gradients is used to compute the first-order partials (gradients) of 'oprice' w.r.t. S,t, and v:
        #   \partial C / \partial S, \prtial C / \partial t, \partial C / \partial v 
        #where 'C' is the oprice.
        #note: partials of K and r are not invoked    
    #2nd order partials are: gamma,charm,vomma 
    greeks_2ndorder_partials = tf.gradients(greeks_1storder_partials[0], [S_,t_,v_])         
        #tf.gradients is now used to compute the second-order partials (gradients) of 'oprice' w.r.t. S,t, and v:
        #note: the mixed partials greeks[1],greeks[2],etc. are not invoked!
    grad += [greeks_1storder_partials,greeks_2ndorder_partials]
    with tf.Session() as sess:      
        value = sess.run(grad,feed_dict={S_: S, K_: K, r_: r, t_: t, v_: v})
        return value     
          
S=49.; K=50.; r=0.05; dt=0.3846; v=0.2 
c = BSMPricer_DL(S,K,r,dt,v); print([c])

