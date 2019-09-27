import numpy as np
import matplotlib.pyplot as plt

class Bandit_d_Thompson_sampling:
    def __init__(self,prior_mean,prior_lambda,tau,gamma,eta):
        self.N=0 #s_hat
        self.gamma=gamma #gamma is the "remember" parameter for used update
        self.eta=eta #eta is the increase factor for UNUSED 
        self.prior_mean=prior_mean
        self.prior_lambda=prior_lambda
        self.posterior_mean=prior_mean
        self.posterior_lambda=prior_lambda
        self.summ=0 #S
        self.tau=tau #by assumption fixed and we give it a value
        #the measure is now the percentage share! the prior mean is 1/n
    
    def pull_sample_mean(self):
        sigma=1/np.sqrt(self.posterior_lambda)
        mean=self.posterior_mean
        #return np.random.random()/np.sqrt(self.posterior_lambda)+self.posterior_mean
        return np.random.normal(mean,sigma) #this is the sampled mean from our known distribution
    
    def update(self,x,used=True):
        tau=self.tau
        gamma=self.gamma
        eta=self.eta
        #the mean is now itself a normally distribuited random variable with
        #mean: posterior_mean and inverse variance lambda_posterior
        #if self.summ!=0:
         #   delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)/self.summ
       # else:
       #     delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)
        delta=gamma
        if used: #if used then x is the gradient value
            self.N = self.N*gamma+1
            self.summ=self.summ*delta+x#mean=(1-1/self.N)*self.mean+1/self.N*x
        else:
            self.N = self.N*gamma
            self.summ=self.summ*gamma
        self.posterior_lambda=self.prior_lambda+tau*self.N#
        self.posterior_mean=(self.summ*tau+self.prior_lambda*self.prior_mean)/self.posterior_lambda
    def correct_percentages(self,used_available_percentage_sum):
        self.posterior_mean=self.posterior_mean/used_available_percentage_sum
def Thompson_sample(n, init=False):
    
    target_percentage=0.9997
    global S
    global coordinate_bandits
    #pick all coordinates who's measures are above 
    prior_mean=1/n#note that prior mean should be slightly above actual values and slightly
    #above critical_measure_value to encourage exploring
    prior_lambda=14000/2*(n/50)**2
    tau=18000
    gamma=0.5#0.78 #remember rate for used
    eta=0.5 #fremember rate for unused
    #initialization------------------------------------------------------------
    if init: # no initialization required
        coordinate_bandits=[]
        for i in range(n): #use discounted thompson sampling and call the objects bandits for fun
            coordinate_bandits.append(Bandit_d_Thompson_sampling(prior_mean,prior_lambda,tau,gamma,eta))
            #1st bandit coresponds to first coordinate and so on
        return
    #end initialization--------------------------------------------------------
    
    #else if we do not initialize-here we just sample, 
    #a different function will be there for update
    # Measure, delta_x, No_current_coords,
    SS=[]
    percentage=np.zeros(n)
    for i in range(n):#sample percentages - do not use the mean directly!
        percentage[i]=coordinate_bandits[i].pull_sample_mean()
        if percentage[i]<0:#if we pull a number below zero
            percentage[i]=0
    total_percentage_sum_for_correction=np.sum(percentage)
    for i in range(n):#correct such that they add up to 100% ie. to 1
        percentage[i]=percentage[i]/total_percentage_sum_for_correction
    #sort the vector
    sorted_nginds = np.argsort(percentage)[::-1]
    #now keep appending until we hit 99.997%  
    current_percentage=0
    i=0
    while current_percentage<target_percentage:
        SS.append(sorted_nginds[i])
        current_percentage+=percentage[sorted_nginds[i]]
        i=i+1
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
    rem_inds = np.setdiff1d(np.arange(n),S) 
    percentage_sum=0
    for coordinate in rem_inds:#update unusued corodinates
        coordinate_bandits[coordinate].update(0,used=False)
        percentage_sum+=coordinate_bandits[coordinate].posterior_mean
    #update the unused percentages/bandits first to check out how much
    norm2=np.linalg.norm(subspace_gradient,ord=2)#extrapolate for the full norm after having updated the unused coords
    norm2_squared=(norm2**2)*1/(1-percentage_sum)
    correcting_percentage_sum=0
    for coordinate in S:
        percentage_update_x=subspace_gradient[coordinate]**2/norm2_squared
        coordinate_bandits[coordinate].update(percentage_update_x,used=True)
        correcting_percentage_sum+=coordinate_bandits[coordinate].posterior_mean
    #now correct all used ones to add up to the correct percentage
    for coordinate in S:
        coordinate_bandits[coordinate].correct_percentages(correcting_percentage_sum/(1-percentage_sum))
    #update used coordinates
   
        
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
      
      
  