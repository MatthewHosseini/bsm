class BSMDLPricer:  
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

    def __init__(self,_S,_K,_r,_t,_v):
        '''Initialize the BSMPricer class:
            @param _S   is the price for the underlying asset
            @param _K   is the strike price for the underlying asset
            @param _r   is the risk-free rate
            @param _dt  is the time to maturity signified by: T-t (annualized)
            @param _v   is the volatility
        '''
        self.S=float(_S)
        self.K=float(_K)
        self.r=float(_r)
        self.t=float(_t)
        self.v=float(_v)
        return     


    def priceanalytic(self):
 
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
          
S=49.; K=50.; r=0.05; t=0.3846; v=0.2 
BSMDLPricer(S,K,r,t,v).priceanalytic()



