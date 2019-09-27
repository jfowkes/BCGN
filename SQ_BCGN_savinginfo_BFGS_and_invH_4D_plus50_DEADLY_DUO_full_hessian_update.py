""" Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update
from trs.trs_approx import trs_approx, trs_approx_precon
from trs.reg import reg, reg_update
from trs.line_search import line_search
from scipy.sparse import csr_matrix
from sampling_funcs import greedy_gauss_southwell, adaptive_greedy_gauss_southwell
import numpy as np
import scipy.linalg as linalg
import math as ma
import pickle
import os

def SQ_BCGN_savinginfo_BFGS_and_invH_4D_plus50_DEADLY_DUO_full_hessian_update(r, J, x0, sampling_func_adaptive, sampling_func, fxopt, it_max, ftol, p, fig, kappa, function_name, run_counter,algorithm='tr'):
    n = x0.size
    S1=np.arange(round(n/2))
    S2=np.setdiff1d(np.arange(n),S1)
    r0=r(x0)
    m= r0.size
    initialized_step_2=False
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
    if kappa==1 and p==2:
        S=greedy_gauss_southwell(n,p,p,init=True)#use p as a dummy grad for init it does not matter
    else:
        S=sampling_func(n,p,init=True)

    k = 0
    x = x0
    delta = None
    #initialize saved metrics-----------------------------------
    if kappa==1:
        max_number_of_gradients=round(n/p)*it_max
    else:
        max_number_of_gradients=round(n/2)*it_max
    #max number of gradients initially defined for saving data ONLY
    #but now is also used used as a termination condition because the budget was
    #amended to not increment when we are at full block size and we just solve TR again
    #sometimes in this case(at full GN-can start from an block size) t
    #he algo just tries but never takes a step
    #for infinite number of iter so we limit it as the budget is not incremented in this case anymore
    #this is now what the budget limit would have been for a non-adaptive scheme
    #when taxing the method at p per iteration even if the blocksize is already max
    max_number_of_gradients=round(max_number_of_gradients*1.5)
    x_k=np.full((n,max_number_of_gradients+1),np.nan)  
    grad=np.full((n,max_number_of_gradients),np.nan) 
    rx_k=np.full((m,max_number_of_gradients+1),np.nan) 
    which_blocks_k=[]#np.full((n,max_number_of_gradients),np.nan) 
    block_size_k=np.full(max_number_of_gradients,np.nan) 
    #grad_s=np.full((n,max_number_of_gradients),np.nan)  #J_s=[]
    norm_grad=np.full(max_number_of_gradients,np.nan)
    norm_grad_s=np.full(max_number_of_gradients,np.nan) 
    objfunction_decrease=np.full(max_number_of_gradients,np.nan)
    J_saved_k=[]
    objfun_value=np.full(max_number_of_gradients+1,np.nan)
    delta_k=np.full(max_number_of_gradients,np.nan)
    steplength_k=np.full(max_number_of_gradients,np.nan)
    budget_saved_k=np.full(max_number_of_gradients+1,np.nan)
    
    budget_saved_k[0]=0
    x_k[:,0]=x0 
    rx_k[:,0]=r(x0)
    objfun_value[0]=f(x0); zeta=0.97
    #end initialization of saved metrics
    actual_previous_useful_k=-1
    ex_step_no=2
    while (not fig and budget < it_max*n and k<max_number_of_gradients) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):
        #n is the dimensionality of the problem. it_max is defined in BCGN.py
        # Randomly select blocks
        #p is how many coordinates are considered            
        
        if steplength_k[k-1]==0:
            if (kappa!=1 or p!=2):
                S, p_in_budget = sampling_func(n,p,step=p,SS=S)
            else:
                if np.linalg.norm(gradf_S,ord=2)<3e-08: #correct failure due to zero grad in that subspace
                    if ex_step_no==1:
                        p_in_budget=len(S2)
                        saved_ex_step_no=1
                    else:
                        p_in_budget=len(S1)
                        saved_ex_step_no=2
                    ex_step_no=0
                    S=np.eye(n)
                else:
                    S=S#np.arange(n)
                    p_in_budget=0
                    
          #  if  actual_previous_useful_k==-1:
           #      actual_previous_useful_k=k-1
        else:
             if kappa==1 and p==2: #if we are on the Gauss Southwell
                if ex_step_no==2: #if we did step in S2 do in S1
                    if k!=0:
                   #     Jtav_2=J(x)[:,len(S1):n]
                  #      old_GN_matrix_2=Jtav_2.T.dot(Jtav_2)
                 #       if np.linalg.matrix_rank(old_GN_matrix_2)==old_GN_matrix_2.shape[0]:
                #            old_inverse_hessian_2=np.linalg.inv(old_GN_matrix_2)
               #         else:
              #              coefficient_n=np.linalg.norm(old_GN_matrix_2,ord=2)/100
             #               old_inverse_hessian_2=np.linalg.inv(old_GN_matrix_2+coefficient_n*np.eye(old_GN_matrix_2.shape[0]))
            
                        old_GN_matrix_2=old_GN_matrix[len(S1):n,len(S1):n]
                        old_inverse_hessian_2=old_inverse_hessian[len(S1):n,:]
                        
                        approx_grad_x_2=previous_grad_2+old_GN_matrix_2.dot(x_k[S2,k]-previous_location_2)
                        #approx_step_2=old_inverse_hessian_2.dot(approx_grad_x_2)
                        approx_step_2=old_inverse_hessian_2.dot(np.hstack((previous_grad_1,previous_grad_2)))+(x_k[S2,k]-previous_location_full[S2])
                     
                        old_step_2=x_k[S2,k]-previous_location_2
                        S_hat = step_and_grad_sampling_4D(approx_step_2, approx_grad_x_2, old_step_2, previous_grad_2)
                        S_hat=np.concatenate((np.zeros([len(S1),S_hat.shape[1]]),S_hat),axis=0)
                        S_1_in_matrix_form=np.eye(n)[0:len(S1)].T
                        S=np.concatenate((S_1_in_matrix_form,S_hat),axis=1)
                        #S= S_1_in_matrix_form
                    else: #on the first step just use the 50 blocks
                        S=np.eye(n)[0:len(S1)].T
                    p_in_budget=S.shape[1]
                    ex_step_no=1
                elif ex_step_no==1: #else if we did step in S1 do in S2
                    if initialized_step_2==True:
                  #      Jtav_1=J(x)[:,0:len(S1)]
                 #       old_GN_matrix_1=Jtav_1.T.dot(Jtav_1)
                #        if np.linalg.matrix_rank(old_GN_matrix_1)==old_GN_matrix_1.shape[0]:
               #             old_inverse_hessian_1=np.linalg.inv(old_GN_matrix_1)
              #          else:
             #               coefficient_n=np.linalg.norm(old_GN_matrix_1,ord=2)/100
            #                old_inverse_hessian_1=np.linalg.inv(old_GN_matrix_1+coefficient_n*np.eye(old_GN_matrix_1.shape[0]))
                        old_GN_matrix_1=old_GN_matrix[0:len(S1),:]
                        old_inverse_hessian_1=old_inverse_hessian[0:len(S1),:]
                        
                        approx_grad_x_1=previous_grad_1+old_GN_matrix_1.dot(x_k[:,k]-previous_location_full)
                        approx_step_1=old_inverse_hessian_1.dot(np.hstack((previous_grad_1,previous_grad_2)))+(x_k[S1,k]-previous_location_full[S1])
                       # approx_step_1=old_inverse_hessian_1.dot(approx_grad_x_1)
                        old_step_1=x_k[S1,k]-previous_location_1
                        S_hat = step_and_grad_sampling_4D(approx_step_1, approx_grad_x_1, old_step_1, previous_grad_1)
                        S_hat=np.concatenate((S_hat,np.zeros([len(S2),S_hat.shape[1]])),axis=0)
                        S_2_in_matrix_form=np.eye(n)[len(S1):n].T
                        #S= S_2_in_matrix_form
                        S=np.concatenate((S_2_in_matrix_form,S_hat),axis=1)
                    else: #on the first step just use the 50 blocks
                        S=np.eye(n)[len(S1):n].T
                    p_in_budget=S.shape[1]
                    ex_step_no=2
                
             else:
                S, p_in_budget = sampling_func(n,p)
           #  actual_previous_useful_k=-1
        #S= sampling_func(n,p,S,save_info_resampling_flag)
        # Assemble block-reduced JACOBIAN matrices and COMPUTE GRADIENT IN
        #THE SUBSPACE OF INTEREST--------------------------------------------
        if len(S.shape)==1:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
      #      if ex_step_no==1:
       #         old_delta=delta
        #        delta=0.01
        #    elif ex_step_no==2: #THE DELTA TRICK
         #       delta=old_delta
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
                S = sampling_func_adaptive(n,step,step=True) #add "step" coordinates to S

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
                if k == 0 and algorithm.startswith('tr') or algorithm == 'reg':
                    delta = linalg.norm(gradf_S)/10
            #put this here because the adaptive was starting with 2D-like TR
            #while full GN was not this limited the adaptive progress
            
                p_in += step #account for considering extra coordinates
                p_in_budget +=step
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
            #record change to the budget
            #print('Iteration:', k, 'max block size:', p_in)
            delta_k[k]=delta #!!!!!!!!!!!!!!!!!
            # Update parameter and take step---------------------------------------
            #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
            #save residual
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
#!!!!!!!!!!!!!#end IF len(S.shape)==1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            J_S = J(x).dot(S)
            if ex_step_no==1:
                J_upgrade=J(x).dot(S[:,0:len(S1)])
            else:
                J_upgrade=J(x).dot(S[:,0:len(S2)])
            rx = r(x)
            gradf_S_upgrade=J_upgrade.T.dot(rx)
            gradf_S = J_S.T.dot(rx)

            # Set initial trust region radius
            if k == 0 and algorithm.startswith('tr') or algorithm == 'reg':
                delta = linalg.norm(gradf_S)/10
                if delta == 0:
                    delta = 1

            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            else:
                raise RuntimeError(algorithm + 'unimplemented!')

        # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            # Iteratively refine block size
            #save residual data before updating x
            # Update parameter and take step
            if algorithm.startswith('tr'):
                x, delta = tr_update_2(f, x, s_S, S, Delta_m, delta)
            else:
                raise RuntimeError(algorithm + 'unimplemented!')
            delta_k[k]=delta #!!!!!!!!!!!!!!!!!
    #save data------part 1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        budget += p_in_budget
        JJ=J(x)
        grad[:,k]=JJ.T.dot(rx)
        norm_grad[k]=max(abs(grad[:,k])) #infinity norm
        if len(S.shape)==1:
            block_size_k[k]=S.size
            norm_grad_s[k]=max(abs(grad[S,k]))
            which_blocks_k.append(S)
            subspace_gradient=np.zeros(n)
            subspace_gradient=grad[S,k]
        else:
            block_size_k[k]=S.shape[1]
            subspace_gradient=S.dot(gradf_S)
            which_blocks_k.append(S)
            norm_grad_s[k]=max(abs(S.dot(gradf_S)))
     
        objfunction_decrease[k]=f(x)-np.linalg.norm(rx, ord=2)
        #note that rx was set to r(x) before x was updated!
        J_saved_k.append(J(x))
        steplength_k[k]=np.linalg.norm(x-x_k[:,k],ord=2)
            
        if initialized_step_2==True and steplength_k[k]!=0 and kappa==1 and p==2: #If the initialization was done: start doing upgrades
            if ex_step_no==1: #if we took a succesful step in S1
                #if np.linalg.norm(x_k[:,k]-x2,ord=2)!=0:#upgrade the hessian in S1
                g1=gradf_S_upgrade
                x1=x_k[:,k]
                grad_current_1=g1
                current_location_1=x1[0:len(S1)]
                grad_S1_change=grad_current_1-previous_grad_1
                location_S1_change=current_location_1-previous_location_1
                previous_grad_1=grad_current_1
                previous_location_1=current_location_1
            elif ex_step_no==2 and np.linalg.norm(x_k[len(S1):n,k]-previous_location_2)!=0: #else if we did a succ step in S2
               g2=gradf_S_upgrade #upgrade the hessian in S2 np.linalg.norm(x_k[:,k]-x1,ord=2)!=0 and 
               x2=x_k[:,k]
               grad_current_2=g2
               current_location_2=x2[len(S1):n]
               grad_S2_change=grad_current_2-previous_grad_2
               location_S2_change=current_location_2-previous_location_2
               previous_grad_2=grad_current_2
               previous_location_2=current_location_2
               previous_location_full=x2
               
               delta_grad_full=np.hstack((grad_S1_change,grad_S2_change))
               delta_location_full=np.hstack((location_S1_change,location_S2_change))
               old_GN_matrix=BFGS_update(old_GN_matrix,delta_grad_full,delta_location_full)
               old_inverse_hessian=BFGS_update(old_inverse_hessian,delta_location_full,delta_grad_full)  
               
            elif ex_step_no==0: #else if we did full GN step and TAKEN the step
               x2=x_k[:,k]
               x1=x_k[:,k]
               grad_current_2=gradf_S[S2]
               grad_current_1=gradf_S[S1]
               current_location_2=x2[len(S1):n]
               current_location_1=x1[0:len(S1)]
               #update both streams and compute changes
               grad_S1_change=grad_current_1-previous_grad_1
               location_S1_change=current_location_1-previous_location_1
               grad_S2_change=grad_current_2-previous_grad_2
               location_S2_change=current_location_2-previous_location_2
               #do update of full hessian
               old_GN_matrix=BFGS_update(old_GN_matrix,delta_grad_full,delta_location_full)
               old_inverse_hessian=BFGS_update(old_inverse_hessian,delta_location_full,delta_grad_full)  
               #set all previous values from current location - we used full space and know all!
               previous_grad_1=grad_current_1
               previous_location_1=current_location_1
               previous_grad_2=grad_current_2
               previous_location_2=current_location_2
               previous_location_full=x2
        
        elif k==0 and kappa==1 and p==2:#initializae the hessian -and also update it
            J1=J_upgrade
            g1=gradf_S_upgrade
            x1=x_k[:,k]
            previous_grad_1=g1
            previous_location_1=x1[0:len(S1)]
        elif ex_step_no==2 and steplength_k[k]!=0 and initialized_step_2==False and kappa==1 and p==2: #initialization part 2
            J2=J_upgrade
            g2=gradf_S_upgrade
            x2=x_k[:,k]
            previous_grad_2=g2
            previous_location_2=x2[len(S1):n]
            
            J_full_est=np.hstack((J1,J2))
            old_GN_matrix=J_full_est.T.dot(J_full_est)
            if np.linalg.matrix_rank(old_GN_matrix)==old_GN_matrix.shape[0]:
                old_inverse_hessian=np.linalg.inv(old_GN_matrix)
            else:
                coefficient_n=np.linalg.norm(old_GN_matrix,ord=2)/100
                old_inverse_hessian=np.linalg.inv(old_GN_matrix+coefficient_n*np.eye(old_GN_matrix.shape[0]))
            
            initialized_step_2=True
            previous_location_full=x2
        elif ex_step_no==0 and steplength_k[k]!=0 and initialized_step_2==False and kappa==1 and p==2: #initialization part 2
           #if we did ful GN at initialization
           previous_location_1=x_k[0:len(S1),k]
           previous_location_2=x_k[len(S1):n,k]
           previous_grad_1=gradf_S[0:len(S1)]
           previous_grad_2=gradf_S[len(S1):n]
           old_GN_matrix=J_S.T.dot(J_S)
           if np.linalg.matrix_rank(old_GN_matrix)==old_GN_matrix.shape[0]:
               old_inverse_hessian=np.linalg.inv(old_GN_matrix)
           else:
                coefficient_n=np.linalg.norm(old_GN_matrix,ord=2)/100
                old_inverse_hessian=np.linalg.inv(old_GN_matrix+coefficient_n*np.eye(old_GN_matrix.shape[0]))
           initialized_step_2=True
           previous_location_full=x_k[:,k]
           saved_ex_step_no=2#put this here to then start from 1st block again after initialization if a FULLGN was done on the very first step
           #end initialization
             #now get back into the usual sequence
        #if steplength_k[k]==0 and we have zero grad norm, move forward without taking the step
        if steplength_k[k]!=0 and ex_step_no==0:
            ex_step_no=saved_ex_step_no
    #end initializae hessian and update it           
        #end save data--------part 1------- 
        k += 1
        #savedata part 2-----------
        rx_k[:,k]=r(x)
        x_k[:,k]=x
        fx=f(x)
        objfun_value[k]=fx
        budget_saved_k[k]=budget
#endsavedata part2----------]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
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
                 pickle_everything(rx_k,function_name,run_counter,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value,delta_k,steplength_k,budget_saved_k)
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
    pickle_everything(rx_k,function_name,run_counter,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value,delta_k,steplength_k,budget_saved_k)
    
    if fig is None:
         #pickle the saved metrics--------------------------------------------
        pickle_everything(rx_k,function_name,run_counter,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value,delta_k,steplength_k,budget_saved_k)
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

def pickle_everything(rx_k,function_name,run_counter,p,n,kappa,x_k,grad,which_blocks_k,block_size_k,norm_grad,norm_grad_s,objfunction_decrease,J_saved_k,objfun_value,delta_k,steplength_k,budget_saved_k):
    save_dir='fct_data'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    if kappa==1:
        if p==2:
            function_name=function_name+'_2D'
        elif p==n:
            function_name=function_name+'_Full_GN'
        else:
            function_name=function_name+'_N_over_2D'
    else:
        function_name=function_name+'_Adaptive'
    function_name=function_name+'_Run'+str(run_counter)
    function_name=save_dir+'/'+function_name
    
    filehandler=open(function_name+'.grad', 'wb')
    pickle.dump(grad, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.rx_k', 'wb')
    pickle.dump(rx_k, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.steplength_k', 'wb')
    pickle.dump(steplength_k, filehandler, protocol=-1)
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

    filehandler=open(function_name+'.delta_k', 'wb')
    pickle.dump(delta_k, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open(function_name+'.budget_saved_k', 'wb')
    pickle.dump(budget_saved_k, filehandler, protocol=-1)
    filehandler.close()
    
def BFGS_update(H,delta_grad,delta_x):
    #H is the HESSIAN not the inferse and this is rather the DFP update than the BFGS
    rho=1/delta_grad.dot(delta_x)
    if np.isnan(rho):
        return H
    bracket1=(np.eye(H.shape[0])-rho*np.outer(delta_grad,delta_x))
    bracket2=(np.eye(H.shape[0])-rho*np.outer(delta_x,delta_grad))
    A=np.dot(bracket1,np.dot(H,bracket2))+rho*np.outer(delta_grad,delta_grad)
    return A

def principal_component_sampling(H,grad):
    eigval, V = np.linalg.eig(H)
    size=len(grad)
    lambda_times_approx_grad_T_v_i=np.zeros(size)
    
    for i in range(size):
        v_i=V[:,i]
        lambda_times_approx_grad_T_v_i[i]=np.abs(eigval[i]*grad.dot(v_i))
    total_norm_squared=np.sum(lambda_times_approx_grad_T_v_i)
    
    #sort!!!! the lambda times bla vector
    sorted_nginds = np.argsort(np.fabs(lambda_times_approx_grad_T_v_i))[::-1]
    i=0
    current_norm_squared=0
    S=np.zeros((size,size))
    while current_norm_squared<0.997*total_norm_squared:
        index=sorted_nginds[i]
        current_norm_squared+=lambda_times_approx_grad_T_v_i[index]
        S[:,i]=V[:,index]
        i=i+1
    return S[:,0:i],V,lambda_times_approx_grad_T_v_i#need all up until i-1!

def principal_component_sampling_increase(S,V,lambda_times_approx_grad_T_v_i):
    size=len(lambda_times_approx_grad_T_v_i)
    Z=np.zeros((size,size))
    current_number_dimensions=S.shape[1]
    current_number_dimensions0=current_number_dimensions
    Z[:,0:current_number_dimensions]=S
    sorted_nginds = np.argsort(np.fabs(lambda_times_approx_grad_T_v_i))[::-1]
    i=1
    while current_number_dimensions<size and i<2:
        index=sorted_nginds[current_number_dimensions]
        Z[:,current_number_dimensions]=V[:,index]
        current_number_dimensions+=1
    budget=current_number_dimensions-current_number_dimensions0
    if current_number_dimensions==size:
        return np.arange(size),V,lambda_times_approx_grad_T_v_i,budget
    return Z[:,0:current_number_dimensions],V,lambda_times_approx_grad_T_v_i,budget

def tr_update_2(f, x, s_S, S, Delta_m, delta):

    # Trust Region parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = 0.5
    GAMMA2 = 2.
    DELTA_MIN = 1e-150
    DELTA_MAX = 1e150

    # Evaluate sufficient decrease
    s = S.dot(s_S)
    rho = (f(x) - f(x+s))/Delta_m

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update trust region radius
    if rho < ETA1:
        delta *= GAMMA1
        delta = max(delta,DELTA_MIN)
    elif rho >= ETA2:
        delta *= GAMMA2
        delta = min(delta,DELTA_MAX)
    elif np.linalg.norm(s,ord=2)>=0.95*delta:
        delta *= 1.1
        delta = min(delta,DELTA_MAX)

    return x, delta

def step_and_grad_sampling_4D (step, grad, old_step, old_grad):
    ssize=len(step)
    if np.linalg.norm(step,ord=2)!=0:
        v1=1/np.linalg.norm(step,ord=2)*step
        index=np.random.choice(np.arange(ssize),size=1,replace=False)
    else:
        v1=np.zeros(ssize)
        index=np.random.choice(np.arange(ssize),size=1,replace=False)
        v1[index]=1
        
    grad_T_v1=grad.dot(v1)
    if np.linalg.norm(grad-grad_T_v1*v1,ord=2)!=0:
        v2=grad-grad_T_v1*v1
        v2=1/np.linalg.norm(v2,ord=2)*v2
    else:
        v2=grad
        rem_inds = np.setdiff1d(np.arange(ssize),index)
        indices=np.random.choice(np.arange(rem_inds),size=int(np.round(ssize/2)),replace=False)
        v2[indices]=v2[indices]-0.1*np.linalg.norm(grad,ord=2)
        v2=1/np.linalg.norm(v2,ord=2)*v2
        
    old_step_T_v1=old_step.dot(v1)
    old_step_T_v2=old_step.dot(v2)
    if np.linalg.norm(old_step-old_step_T_v1*v1-old_step_T_v2*v2,ord=2)!=0:
        v3=old_step-old_step_T_v1*v1-old_step_T_v2*v2
        v3=1/np.linalg.norm(v3,ord=2)*v3
    else:
        v3=old_step
        rem_inds = np.setdiff1d(np.arange(ssize),index)
        indices=np.random.choice(rem_inds,size=int(np.round(ssize/3)),replace=False)
        v3[indices]=v3[indices]-0.1*np.linalg.norm(old_step+0.1,ord=2)
        v3=1/np.linalg.norm(v3,ord=2)*v3
    old_grad_T_v1=old_grad.dot(v1)
    old_grad_T_v2=old_grad.dot(v2)
    old_grad_T_v3=old_grad.dot(v3)
    if np.linalg.norm(old_grad-old_grad_T_v1*v1-old_grad_T_v2*v2-old_grad_T_v3*v3,ord=2)!=0:
        v4=old_grad-old_grad_T_v1*v1-old_grad_T_v2*v2-old_grad_T_v3*v3
        v4=1/np.linalg.norm(v4,ord=2)*v4
    else:
        v4=old_grad
        rem_inds = np.setdiff1d(np.arange(ssize),index)
        indices=np.random.choice(rem_inds,size=int(np.round(ssize/1.5)),replace=False)
        v4[indices]=v4[indices]-0.1*np.linalg.norm(old_grad+0.1,ord=2)
        v4=1/np.linalg.norm(v4,ord=2)*v4 
    
    S=np.array([v1,v2,v3,v4])
    S=S.T
    return S

def step_and_grad_sampling_grow(S):
    return np.arange(S.shape[0]),S.shape[0]-2