import numpy as np
import matplotlib.pyplot as plt

class Bandit_d_Thompson_sampling:
    def __init__(self,prior_mean,prior_lambda,tau,gamma):
        self.N=0 #s_hat
        self.gamma=gamma
        self.prior_mean=prior_mean
        self.prior_lambda=prior_lambda
        self.posterior_mean=prior_mean
        self.posterior_lambda=prior_lambda
        self.summ=0 #S
        self.tau=tau #by assumption fixed and we give it a value
    
    def pull_sample_mean(self):
        sigma=1/np.sqrt(self.posterior_lambda)
        mean=self.posterior_mean
        #return np.random.random()/np.sqrt(self.posterior_lambda)+self.posterior_mean
        return np.random.normal(mean,sigma) #this is the sampled mean from our known distribution
    
    def update(self,x,used=True):
        tau=self.tau
        gamma=self.gamma
        #the mean is now itself a normally distribuited random variable with
        #mean: posterior_mean and inverse variance lambda_posterior
        #if self.summ!=0:
         #   delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)/self.summ
       # else:
       #     delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)
        delta=gamma
        if used:
            self.N = self.N*gamma+1
            self.summ=self.summ*delta+x#mean=(1-1/self.N)*self.mean+1/self.N*x
        else:
            self.N = self.N*gamma
            self.summ=self.summ*delta
        self.posterior_lambda=self.prior_lambda+tau*self.N#
        self.posterior_mean=(self.summ*tau+self.prior_lambda*self.prior_mean)/self.posterior_lambda

def Thompson_sample(n, init=False):
    
    critical_measure_value=0.15#we do not take coordinates with measure below this value
    #can make it time dependent i.e. iteration dependent later if needed.
    global S
    global coordinate_bandits
    #pick all coordinates who's measures are above 
    prior_mean=17#note that prior mean should be slightly above actual values and slightly
    #above critical_measure_value to encourage exploring
    prior_lambda=0.1
    tau=1
    gamma=0.75#0.78
    if init: # no initialization required
        coordinate_bandits=[]
        for i in range(n): #use discounted thompson sampling and call the objects bandits for fun
            coordinate_bandits.append(Bandit_d_Thompson_sampling(prior_mean,prior_lambda,tau,gamma))
            #1st bandit coresponds to first coordinate and so on
        return
    
    #else if we do not initialize-here we just sample, 
    #a different function will be there for update
    # Measure, delta_x, No_current_coords,
    SS=[]
    for i in range(n):
        if critical_measure_value<coordinate_bandits[i].pull_sample_mean():#if the value I'm sampling is above the critical mean, include this coordinate
            SS.append(i)
    if len(SS)==1:
        SS=np.array(SS)
        rem_inds = np.setdiff1d(np.arange(n),SS) 
        SA = np.random.choice(rem_inds,size=1,replace=False)
        SS = np.append(SS,SA) # and we now have 2 coords
            #SS = np.hstack((SS,SA))
    elif len(SS)==0:
        SS = np.random.choice(np.arange(n),size=2,replace=False)
    else:
        SS=np.array(SS)
    S=SS
    return S

def update_all_coordinates(subspace_gradient,S,n):
    #note that this can be any other measure but we keep the same name here
    subspace_gradient=abs(subspace_gradient)
    global coordinate_bandits
    #update used coordinates
    #print(len(S))
    #print(type(S[0]))
    for coordinate in S:
        coordinate_bandits[coordinate].update(subspace_gradient[coordinate],used=True)
    #update used coordinates
    rem_inds = np.setdiff1d(np.arange(n),S) 
    for coordinate in rem_inds:#update unusued corodinates
        coordinate_bandits[coordinate].update(0,used=False)
        
if __name__ == '__main__':
  NoITER=20 
  axis=np.array(np.arange(NoITER))
  c1_grad= np.sin(axis)
  c2_grad= np.sin(axis+3)
  c3_grad= np.sin(axis+7)
  c4_grad= np.sin(axis+10)
  c5_grad= np.sin(axis+14)
  c_grad=np.vstack([c1_grad,c2_grad,c3_grad,c4_grad,c5_grad])
  n=5#assume we start with full GN
  Thompson_sample(n, init=True)
  whichcoord=np.zeros([n,NoITER])
  #whichcoord[:,0]=np.ones(n)
  for i in range(NoITER):
      S=Thompson_sample(n)
      #take iteration,compute gradient in subspace
      #update coordinate_bandits
      current_grad=np.zeros(n)
      current_grad[S]=c_grad[S,i]
      update_all_coordinates(-current_grad,S,n)
      #remember which coords were updated
      whichcoord[S,i]=np.ones(len(S))
      
  plt.figure(figsize=(12,7))
  plt.clf()
  plt.subplot(2,1,1)
  plt.plot(c1_grad, label='1st coord grad')
  plt.plot(c2_grad, label='2nd coord grad')
  plt.plot(c3_grad, label='3rd coord grad')
  plt.plot(c4_grad, label='4th coord grad')
  plt.plot(c5_grad, label='5th coord grad')
  plt.plot(np.ones(NoITER)*0.1, label='critical Measure Value')
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(whichcoord[0,:],marker='o',label='1st Coord')
  plt.plot(2*whichcoord[1,:],marker='x',label='2 Coord')
  plt.plot(3*whichcoord[2,:],marker='v',label='3rd Coord')
  plt.plot(4*whichcoord[3,:],marker='s',label='4th Coord')
  plt.plot(5*whichcoord[4,:],marker='P',label='5th Coord')
  plt.legend()
      
      
  