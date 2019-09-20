import os
path = os.getcwd()
os.chdir('/Users/mhosseini/Documents/projects/book/nn/code/python')
figdir = '/Users/mhosseini/Documents/projects/book/figs/'
       
def SimulateStockPaths(S, alpha, delta, sigma, T, N, n):
    ''' Generate Geometric Brownian Motion paths '''
    import random
    import numpy as np
    random.seed(12345)
    h = T/n
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))

def PlotStocks():
    ''' Plot simulated stock paths '''
    import matplotlib.pyplot as plt 
    T = 0.5 #years of training data
    days = int(250*T)
    path1 = SimulateStockPaths(49., .05, 0, .20, T, 1, days) 
    path2 = SimulateStockPaths(49., .05, 0, .20, T, 1, days) 
    #plot stock paths
    plt.plot(path1, 'k-', label = 'GBM: Training')
    plt.plot(path2, 'k-.', label = 'GBM: Test')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.ylabel(r'Stock Prices ($)')
    plt.xlabel(r'no. of days')
    plt.savefig(figdir + '/GBM.png')
    plt.show()
 
def PriceUsingAnalytic():
    import numpy as np
    from BSMClass import BSMPricer
    import matplotlib.pyplot as plt   
    
    S=49.; K=50.; r=0.05; t=0.3846; v=0.2 
    #get vanilla option prices:
    c = BSMPricer(S,K,r,t,v,1).priceanalytic()
    p = BSMPricer(S,K,r,t,v,-1).priceanalytic()
    print(r"call= $%.4f,put = $%.4f" % (c,p))
    #get related sensitivities:
    dc=BSMPricer(S,K,r,t,v,1).delta()
    dp=BSMPricer(S,K,r,t,v,-1).delta()
    g=BSMPricer(S,K,r,t,v,1).gamma()
    print(r"delta_{call}= %.4f,delta_{put} = %.4f,gamma= %.4f" % (dc,dp,g))
    tc=BSMPricer(S,K,r,t,v,1).theta()
    tp=BSMPricer(S,K,r,t,v,-1).theta()
    tt=BSMPricer(S,K,r,t,v,1).charm()
    print(r"theta_{call}= %.4f,theta{put} = %.4f,charm= %.4f" % (tc,tp,tt))
    v=BSMPricer(S,K,r,t,v,1).vega()
    vv=BSMPricer(S,K,r,t,v,1).vomma()
    print(r"vega = %.4f,vomma= %.4f" % (v,vv))
    #print a summary table of option prices and their sensitivities:
    import pandas
    option = ["call (c)","put (p)"]
    sensitivity = ["price","delta_c", "delta_p", "gamma", "theta_c", "theta_p", "charm","vega","vomma"]
    data = np.array([[round(c,4), round(dc,4), 0, round(g,4), round(tc,4), 0, round(tt,4), round(v,4), round(vv,4)],
                     [round(p,4), 0, round(dp,4), round(g,4), 0, round(tp,4), round(tt,4), round(v,4), round(vv,4)]] )
    xl_table=pandas.DataFrame(data, option, sensitivity)
    print(xl_table)
    #create datapoints 
    S=100.; K=50.; r=0.05; t=0.3846; v=0.2 
    S = np.arange(10, 100)   #range of stock prices
    c_dp = [BSMPricer(x, K, r, t, v, 1).priceanalytic() for x in S]   
    p_dp = [BSMPricer(x, K, r, t, v, -1).priceanalytic() for x in S]
    #plot vanilla option prices
    plt.plot(S,c_dp, 'k', label = "Call")
    plt.plot(S,p_dp, 'k-.', label  = "Put")
    plt.legend(frameon=False)
    plt.ylabel(r'Stock Price, $S$')
    plt.xlabel(r'Option Price, $c$')
    plt.tight_layout()
    plt.savefig(figdir + '/OPrices.png')
    plt.show()
    
    S = np.arange(1, 100)   #range of stock prices 
    k = [45,50,55]          #range of strike prices 
    r=0.05 ; t=0.3846 ; v=0.2
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(15, 15))
    ccols = ['r', 'r', 'r']
    clines=[':','-','--']
    pcols = ['g', 'g', 'g']
    plines=[':','-','--']
    cols = ['gray', 'gray', 'gray']
    lines=[':','-','--']
    plt.subplot(321)
    j=0
    for i in k:
        delta_c = [BSMPricer(x,i,r,t,v,1).delta() for x in S]
        delta_p = [BSMPricer(x,i,r,t,v,-1).delta() for x in S]
        plt.plot(delta_c, label = (r"$\delta_{call}$ at K=%i" % i ), linestyle= clines[j], color=ccols[j])
        plt.plot(delta_p,label = (r"$\delta_{put}$ at  K=%i" % i ), linestyle= plines[j], color=pcols[j])
        j+=1
    plt.legend(frameon=False)
    plt.ylabel(r'$\delta$')
    plt.xlabel(r'$S_t$')
    
    plt.subplot(322)
    j=0
    for i in k:
        gamma = [BSMPricer(x,i,r,t,v,1).gamma() for x in S]
        plt.plot(gamma, label = (r"$\gamma$ at K=%i" % i ), linestyle= lines[j], color=cols[j])
        j+=1
    plt.ylabel(r'$\gamma=\frac{\partial^2 \delta}{\partial S^2}$')
    plt.legend(frameon=False)
    plt.xlabel(r'$S_t$')
    
    plt.subplot(323)
    j=0
    for i in k:
        theta_c = [BSMPricer(x,i,r,t,v,1).theta() for x in S]
        theta_p = [BSMPricer(x,i,r,t,v,-1).theta() for x in S]
        plt.plot(theta_c, label = (r"$\theta_c$ at K=%i" % i ), linestyle= clines[j], color=ccols[j])
        plt.plot(theta_p, label = (r"$\theta_p$ at K=%i" % i ), linestyle= plines[j], color=pcols[j])
        j+=1
    plt.ylabel(r'$\theta$')
    plt.legend(frameon=False)
    plt.xlabel(r'$S_t$')
    
    plt.subplot(324)
    j=0
    for i in k:
        charm = [BSMPricer(x,i,r,t,v,1).charm() for x in S]
        plt.plot(charm, label = (r"$charm$ at K=%i" % i ), linestyle= lines[j], color=cols[j])
        j+=1
    plt.ylabel(r'$charm=\frac{\partial^2 \theta}{\partial S^2}$')
    plt.legend(frameon=False)
    plt.xlabel(r'$S_t$')
    
    plt.subplot(325)
    j=0
    for i in k:
        vega = [BSMPricer(x,i,r,t,v,1).vega() for x in S]
        plt.plot(vega, label = (r"$\nu$ at K=%i" % i ), linestyle= lines[j], color=cols[j])
        j+=1
    plt.legend(frameon=False)
    plt.ylabel(r'$\nu = \frac{\partial V}{\partial \sigma}$')
    plt.xlabel(r'$S_t$')
    
    plt.subplot(326)
    j=0
    for i in k:
        vomma = [BSMPricer(x,i,r,t,v,1).vomma() for x in S]
        plt.plot(vomma, label = (r"$vomma$ at K=%i" % i ), linestyle= lines[j], color=cols[j])
        j+=1
    plt.legend(frameon=False)
    plt.ylabel(r'$vomma=\frac{\partial^2 \nu}{\partial \sigma^2}$')
    plt.xlabel(r'$S_t$')
    
    plt.savefig(figdir + '/OSensitivities.png')
    plt.show()

    
def main():
    PlotStocks()
    PriceUsingAnalytic()
    #PriceUsingMonteCarlo()
    
if __name__ == "__main__":
    main()



#the following will go before def main() in the BSMMain.py
    
def PriceUsingMonteCarlo():
    from BSMClass import BSMPricer
    S=49.; K=50.; r=0.05; t=0.3846; v=0.2 
    #get vanilla option prices:
    c = BSMPricer(S,K,r,t,v,1).pricemcs()
    p = BSMPricer(S,K,r,t,v,-1).pricemcs()
    print('Option Prices (c= %.4f,p= %.4f)' % (c,p))


#the following will go at the end of the BSMClass.py
    
    def pricemcs(self):
        '''
        Price using Monte-Carlo Simulation:           
            Using MCS with 50,000 simulation yields (c= 2.3974,p= 2.4406)
        '''
        def getstockprice(S,r,t,v):
            from random import gauss
            return S * exp((r - 0.5 * v**2) * t + v * sqrt(t) * gauss(0,1.0))
    
        def getpayoff(S,K,f):
            return max(f*(S - K),0.0)

        simulations = 50000
        payoffs = 0.0
        for i in range(simulations):
            S_ = getstockprice(self.S,self.r,self.t,self.v)
            payoffs += getpayoff(S_, self.K,self.f)
        
        return exp(-self.r * self.t) * payoffs / float(simulations)
