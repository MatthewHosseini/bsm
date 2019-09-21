from scipy import stats
from math import log,sqrt,exp
class BSMPricer:  
    ''' 
    Author: Matthew Hosseini
    Date:   July 31, 2019
    Guide:
        
    Using the book by John C. Hull on Options and Derivatives to: 
        
        1) price vanilla options, and
        2) compute sensitivities. 

    The price and related sensitivities are computed by: 
        
        1) initializing the required parameters:
            S=49.; K=50.; r=0.05 (5%); dt=0.3846 (20/52 weeks); v=0.2 (20%)
        2) invoking price() function:
            c = BSMPricer(S,K,r,t,v,1).price()
            p = BSMPricer(S,K,r,t,v,-1).price()
        3) and related 1st and 2nd order derivatives (sensitivities) 
            dc = BSMPricer(S,K,r,t,v,1).delta()
            dp = BSMPricer(S,K,r,t,v,-1).delta()
            g = BSMPricer(S,K,r,t,v,1).gamma()
    '''

    def __init__(self,_S,_K,_r,_t,_v,_f):
        '''Initialize the BSMPricer class:
            @param _S   is the price for the underlying asset
            @param _K   is the strike price for the underlying asset
            @param _r   is the risk-free rate
            @param _dt  is the time to maturity signified by: T-t (annualized)
            @param _v   is the volatility
            @param _f   is the option type: +/-1 for a call/put option
        '''
        self.S=float(_S)
        self.K=float(_K)
        self.r=float(_r)
        self.t=float(_t)
        self.v=float(_v)
        self.f=float(_f)
        return     
    
    def d1(self):
        '''
        d1 is a conditional probability at maturity, signifying: 
            1) direction of  a spot price rising (declining) above (below) a strike price
            2) magnitude of the difference between spot and strike prices         
        '''
        return (log(self.S / self.K) + (self.r+0.5*self.v**2)*self.t) / (self.v*sqrt(self.t))
    
    def d2(self):
        '''
        d2 is the probability that the option will expire in the money (spot rises above strike for a call)
        '''
        return self.d1() - self.v * sqrt(self.t)
        #return (log(self.S / self.K) + (self.r-0.5*self.v**2)*self.t) / (self.v*sqrt(self.t))
    
    def priceanalytic(self):
        ''' Price using Closed-Form analytic '''
        d1 = self.d1()
        d2 = self.d2()
        N = stats.norm.cdf
        return self.f*(self.S*N(self.f*d1) - self.K*exp(-self.r*self.t)*N(self.f*d2))   
    
    def delta(self):
        ''' The change in option price given a change in the stock price '''
        N = stats.norm.cdf
        d1 = self.d1()
        return self.f*exp(-self.t)*N(self.f*d1)
    
    def gamma(self):
        ''' The derivative (convexity) of delta '''
        N_ = stats.norm.pdf
        d1 = self.d1()
        return (N_(d1)) / (self.S * self.v * sqrt(self.t))
    
    def theta(self):
        ''' The change in option price given passage (decay) of time '''
        d1 = self.d1()
        d2 = self.d2()
        N = stats.norm.cdf
        N_ = stats.norm.pdf
        return -(self.S * N_(d1) * self.v / (2 * sqrt(self.t))) - self.f* (self.r * self.K * exp(-self.r*self.t) * N(self.f*d2))
      
    def charm(self):
        ''' The derivative (convexity) of theta '''
        d1 = self.d1()
        d2 = self.d2()
        N_ = stats.norm.pdf
        return -(N_(d1)*(2*self.r*self.t) - d2*self.v*sqrt(self.t)) / (2*(self.t)*self.v*sqrt(self.t))

    def vega(self):
        ''' The change in option price given volatility '''
        N_ = stats.norm.pdf
        d1 = self.d1()
        return self.S*N_(d1)*sqrt(self.t)    

    def vomma(self):
        ''' The derivative (convexity) of vega '''
        N_ = stats.norm.pdf
        d1 = self.d1()
        d2 = self.d2()
        return -exp(-self.r*self.t) * d2 / self.v * N_(d1)
