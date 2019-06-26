""" Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update
from trs.trs_approx import trs_approx, trs_approx_precon
from trs.reg import reg, reg_update
from trs.line_search import line_search
from scipy.sparse import csr_matrix
import numpy as np
import scipy.linalg as linalg
import math as ma
import pickle

def RBCGN(r, J, x0, sampling_func, fxopt, it_max, ftol, p, fig, kappa, function_name, algorithm='tr'):
    n = x0.size

    # Adaptive BCGN step size
    STEP = 5

    # Full function and gradient
    def f(z): return 0.5 * np.dot(r(z), r(z)) # f=r^T * r
    def gradf(z): return J(z).T.dot(r(z)) #J^T * r

    # Plotting
    if fig is not None:
        plot_data = np.full((3,it_max+1),np.nan)
        plot_data[0,0] = f(x0)-fxopt
        plot_data[1,0] = linalg.norm(gradf(x0))

    # Metrics
    budget = 0
    tau_budget = np.full(4,np.nan)

    # Initialize block sampling function
    sampling_func(n,p,init=True)

    k = 0
    x = x0
    delta = None
    #initialize saved metrics-----------------------------------
    if kappa==1:
        max_number_of_gradients=round(n/p)*it_max
    else:
        max_number_of_gradients=round(n/2)*it_max
            
    x_k=np.full((n,max_number_of_gradients),np.nan)      
    grad=np.full((n,max_number_of_gradients),np.nan) 
    which_blocks_k=[]#np.full((n,max_number_of_gradients),np.nan) 
    block_size_k=np.full(max_number_of_gradients,np.nan) 
    #grad_s=np.full((n,max_number_of_gradients),np.nan)  #J_s=[]
    norm_grad=np.full(max_number_of_gradients,np.nan)
    norm_grad_s=np.full(max_number_of_gradients,np.nan) 
    objfunction_decrease=np.full(max_number_of_gradients,np.nan)
    J_saved_k=[]
    objfun_value=np.full(max_number_of_gradients,np.nan)
    #end initialization of saved metrics
    
    while (not fig and budget < it_max*n) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):
        #n is the dimensionality of the problem. it_max is defined in BCGN.py
        # Randomly select blocks
        S = sampling_func(n,p) #p is how many coordinates are considered

        # Assemble block-reduced JACOBIAN matrices and COMPUTE GRADIENT IN
        #THE SUBSPACE OF INTEREST--------------------------------------------
        if 'tr_approx' in algorithm: # sparse
            U_S = csr_matrix((np.ones(len(S)),(S,range(len(S)))),shape=(n,len(S)))
            J_S = J(x).dot(U_S) #extracting the relevant columns of the jacobian.
            J_ST = J_S.T.tocsr() #transpose it and convert it to sparse 
            rx = r(x)
            gradf_S = J_ST.dot(rx)
        else: # dense
            J_S = J(x)[:,S]
            rx = r(x)
            gradf_S = J_S.T.dot(rx)
        #--------------------------------------------------------------------
        
        # Set initial trust region radius-----------------------------------
        if k == 0 and algorithm.startswith('tr') or algorithm == 'reg':
            delta = linalg.norm(gradf_S)/10
            if delta == 0:
                delta = 1
        #--------------------------------------------------------------------
        
        # Debug output
        #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

        # Solve subproblem---------------------------------------------------
        if algorithm == 'tr':
            s_S = trs(J_S, gradf_S, delta)
        elif algorithm == 'tr_approx':
            s_S = trs_approx(J_S, J_ST, gradf_S, delta)
        elif algorithm == 'tr_approx_precon':
            s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
        elif algorithm == 'reg':
            s_S, delta = reg(J_S, gradf_S, delta)
        else:
            s_S, delta = line_search(f, x, S, J_S, gradf_S)
        #------------------------------------------------------------------

        # Loop tolerance---this is just for Adaptive - so could have an if-----
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = -Delta_m + kappa*delta*delta > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta
        #-------------------------------------------------------------------

        # Iteratively refine block size=======Adaptive Block Size=============
        p_in = len(S)
        while kappa != 1 and p_in != n and stopping_rule:

            # Increase block size
            step = min(STEP,n-p_in)
            #print('Increasing block size to:',p_in+step)
            S = sampling_func(n,step,step=True) #add "step" coordinates to S

            #2 Assemble block-reduced JACOBIAN matrices and COMPUTE GRADIENT IN
            #THE SUBSPACE OF INTEREST--------------------------------------------
            if 'tr_approx' in algorithm: # sparse
                U_S = csr_matrix((np.ones(len(S)),(S,range(len(S)))),shape=(n,len(S)))
                J_S = J(x).dot(U_S)
                J_ST = J_S.T.tocsr()
                gradf_S = J_ST.dot(rx)
            else: # dense
                J_S = J(x)[:,S]
                gradf_S = J_S.T.dot(rx)
            #------------------------------------------------------------------
            
            p_in += step #account for considering extra coordinates
            
            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            #2 Solve subproblem------------------------------------------------
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            elif algorithm == 'tr_approx':
                s_S = trs_approx(J_S, J_ST, gradf_S, delta)
            elif algorithm == 'tr_approx_precon':
                s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
            elif algorithm == 'reg':
                s_S, delta = reg(J_S, gradf_S, delta)
            else:
                s_S, delta = line_search(f, x, S, J_S, gradf_S)
            #-----------------------------------------------------------------

            #2 Loop tolerance-----------stopping rule for adaptive size------
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
            #stopping_rule = -Delta_m + kappa*delta*delta > 0
            #stopping_rule = linalg.norm(gradf_S) > kappa*delta
            #-----------------------------------------------------------------
       #=======================================================================   '''      
        budget += p_in #record change to the budget
        #print('Iteration:', k, 'max block size:', p_in)

        # Update parameter and take step---------------------------------------
        #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
        if algorithm.startswith('tr'):
            x, delta = tr_update(f, x, s_S, S, gradf_S, Delta_m, delta)
        elif algorithm == 'reg':
            x, delta = reg_update(f, x, s_S, S, Delta_m, delta) # same as tr_update with grow/shrink swapped
        else:
            s = np.zeros(n)
            s[S] = s_S
            x = x + delta*s #delta is now the steplength as overwritten by tr_update
            #it appears that delta is also the next-step TR radius
        #---------------------------------------------------------------------
        #save data
        x_k[:,k]=x   
        JJ=J(x)
        grad[:,k]=JJ.T.dot(rx)
        which_blocks_k.append(S)
        block_size_k[k]=S.size
        norm_grad[k]=max(abs(grad[:,k])) #infinity norm
        norm_grad_s[k]=max(abs(grad[S,k]))
        objfunction_decrease[k]=+f(x)-np.linalg.norm(rx, ord=2)
        #note that rx was set to r(x) before x was updated!
        J_saved_k.append(J(x))
        objfun_value[k]=f(x)
        #end save data 
        k += 1
               # function decrease metrics------------------------------------------
        if fig is None: #tau_budget carries the bugdet IFF the problem was solved
            for itau, tau in enumerate([1e-1,1e-3,1e-5,1e-7]):
                if np.isnan(tau_budget[itau]) and f(x) <= tau*f(x0): # FUNCTION DECREASE CONDTION AS OPPOSED TO GRADIENT ~OLD
                    #print(f(x))
                    #print(f(x0))
                #np.linalg.norm(gradf(x)) <= tau*np.linalg.norm(gradf(x0)): - old condition...
                    tau_budget[itau] = budget
            if np.all(np.isfinite(tau_budget)): # Stop if all function decrease metrics satisfied
                #pickle the saved metrics--------------------------------------------
                 pickle_everything(function_name,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value)
                 return tau_budget
                 
        else: # plotting
            plot_data[0,k] = f(x)-fxopt
            plot_data[1,k] = linalg.norm(gradf(x))
            plot_data[2,k] = p_in
        #----------------------------------------------------------------------
        #end loop

    # Debug output
    #monitor(k, r, x, f, delta, algorithm, gradf)
    #end piclkling saved metrixs
    # Return function decrease metrics (some unsatisfied)
    pickle_everything(function_name,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value)
        
    if fig is None:
         #pickle the saved metrics--------------------------------------------
        pickle_everything(function_name,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value)
        return tau_budget 
    else: # plotting
        return plot_data
    
""" Output Monitoring Information """
def monitor(k, r, x, f, delta, algorithm, gradf, gradf_S=None):

    print('++++ Iteration', k, '++++')
    if algorithm.startswith('tr'):
        print('delta: %.2e' % delta)
    elif algorithm == 'reg':
        print('sigma: %.2e' % delta)
    elif delta is not None:
        print('alpha: %.2e' % delta)

    nr = linalg.norm(r(x))
    ng = linalg.norm(gradf(x))
    nJrr = ng / nr
    if gradf_S is not None:
        ng_S = linalg.norm(gradf_S)
        nJ_Srr = ng_S / nr

    print('x:', x, 'f(x):', f(x))
    print('||r(x)||: %.2e' % nr, '||gradf(x)||: %.2e' % ng,end='')
    if  gradf_S is not None: print('||gradf_S(x)||: %.2e' % ng_S)
    print("||J'r||/||r||: %.2e" % nJrr,end='')
    if gradf_S is not None: print("||J_S'r||/||r||: %.2e" % nJ_Srr)

    if gradf_S is None: print()

def pickle_everything(function_name,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value):
    
    if kappa==1:
        if p==2:
            function_name=function_name+'_2D'
        elif p==n:
            function_name=function_name+'_Full_GN'
        else:
            function_name=function_name+'_N_over_2D'
    else:
        function_name=function_name+'_adaptive'
    
    filehandler=open(function_name+'.grad', 'wb')
    pickle.dump(grad, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.x_k', 'wb')
    pickle.dump(x_k, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.objfun_value', 'wb')
    pickle.dump(objfun_value, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.which_blocks_k', 'wb')
    pickle.dump(which_blocks_k, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.block_size_k', 'wb')
    pickle.dump(block_size_k, filehandler, protocol=-1)
    filehandler.close()

    filehandler=open(function_name+'.norm_grad', 'wb')
    pickle.dump(norm_grad, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.norm_grad_s', 'wb')
    pickle.dump(norm_grad_s, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.objfunction_decrease', 'wb')
    pickle.dump(objfunction_decrease, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.J_k', 'wb')
    pickle.dump(J_saved_k, filehandler, protocol=-1)
    filehandler.close()

        
    
    
    