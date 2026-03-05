import torch 
import numpy as np
import yaml 
from fcts.utils import *
from scipy.special import logit

loaded_parameters = load_objects_from_yaml('trained_parameters.yaml')

n1 = loaded_parameters['n1']
n2 = loaded_parameters['n2']
nq = loaded_parameters['nq']
nl = loaded_parameters['nl']

indices_p = np.array(loaded_parameters['indices_p'])
indices_n = np.array(loaded_parameters['indices_n'])
indices_zeros = np.array(loaded_parameters['indices_zeros'])
device = loaded_parameters['device']
device2 = loaded_parameters['device2']

inv_softplus=  lambda x: x + np.log(-np.expm1(-x))
shrink_simplex_internal= (lambda p: 1 - p[:, :-1] / np.cumsum(p[:, ::-1], axis=1)[:, :0:-1])
shrinkpow= lambda s: np.exp((np.arange(s.shape[1], 0, -1).reshape((1, -1))) * np.log(s))
shrink_simplex= lambda p: shrinkpow(shrink_simplex_internal(p))


def return_all_params(filepath): 
    loaded_parameters = load_objects_from_yaml(filepath)
    theta = np.concatenate(
         (
              ((loaded_parameters['mu_un'])[0][0],),
              (inv_softplus(loaded_parameters['sigma_sq_a'][0][0]),),
              (inv_softplus(loaded_parameters['sigma_sq_b'][0][0]),),
              (inv_softplus(loaded_parameters['sigma_sq_p'][0][0]),),
              (inv_softplus(loaded_parameters['sigma_sq_q'][0][0]),),
              logit(shrink_simplex(np.array(loaded_parameters['alpha_1']).T).flatten()),
              logit(shrink_simplex(np.array(loaded_parameters['alpha_2'])).flatten()),
              logit(np.array(loaded_parameters['pi']).reshape(nq,nl).flatten()),
         )
    )

    gamma = np.concatenate(
        (
            np.array(loaded_parameters['nu_a']).flatten(), 
            inv_softplus(np.array(loaded_parameters['rho_a']).flatten()), 
            np.array(loaded_parameters['nu_b']).flatten(),
            inv_softplus(np.array(loaded_parameters['rho_b']).flatten()),
            np.array(loaded_parameters['nu_p']).flatten(),
            inv_softplus(np.array(loaded_parameters['rho_p']).flatten()),
            np.array(loaded_parameters['nu_q']).flatten(),
            inv_softplus(np.array(loaded_parameters['rho_q']).flatten()),
            logit(shrink_simplex(np.array(loaded_parameters['tau_1'])).flatten()),
            logit(shrink_simplex(np.array(loaded_parameters['tau_2'])).flatten()),
        )
    )

    assert len(theta.shape) == 1
    assert theta.shape[0] == 5 + nq - 1 + nl - 1 + nq * nl
    assert len(gamma.shape) == 1
    assert gamma.shape[0] == 4 * n1 + 4 * n2 + (n1 * (nq - 1)) + (
        n2 * (nl - 1)
    )

    return np.concatenate((gamma, theta))

def expectation_loglike_Y1(tau_1, alpha_1):
    return tau_1.sum(0) @ torch.log(alpha_1)

def expectation_loglike_Y2(tau_2, alpha_2):
    return tau_2.sum(0) @ torch.log(alpha_2).t()

def expectation_loglike_A(nu_a, rho_a, sigma_sq_a):
        return -n1 / 2 * (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))+ torch.log(sigma_sq_a)) - 1 / (2 * sigma_sq_a) * torch.sum(rho_a + nu_a ** 2)

def expectation_loglike_B(nu_b, rho_b, sigma_sq_b):
    return -n1 / 2 * (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))+ torch.log(sigma_sq_b)) - 1 / (2 * sigma_sq_b) * torch.sum(rho_b + nu_b ** 2)

def expectation_loglike_P(nu_p, rho_p, sigma_sq_p):
    return -n2 / 2 * (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))+ torch.log(sigma_sq_p)) - 1 / (2 * sigma_sq_p) * torch.sum(rho_p + nu_p ** 2)

def expectation_loglike_Q(nu_q, rho_q, sigma_sq_q):
    return -n2 / 2 * (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))+ torch.log(sigma_sq_q)) - 1 / (2 * sigma_sq_q) * torch.sum(rho_q + nu_q ** 2)



def entropy_rx(rho_a, rho_b, rho_p, rho_q, tau_1, tau_2):
    tau_i = tau_1.clone()
    tau_i[torch.where(tau_i==0)]=1e-6

    return (1/2* (2 * n1 + 2 * n2)* 
        (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))+ 1)
        + 1 / 2 * torch.sum(torch.log(rho_a))
        + 1 / 2 * torch.sum(torch.log(rho_b))
        + 1 / 2 * torch.sum(torch.log(rho_p))
        + 1 / 2 * torch.sum(torch.log(rho_q))
        - torch.sum(tau_i * torch.log(tau_i))
        - torch.sum(tau_2 * torch.log(tau_2)))


def expectation_loglike_X_cond_ABPY1Y2(
        mu_un,nu_a,nu_b,nu_p,nu_q,rho_a,rho_b,rho_p,rho_q,tau_1,tau_2,pi):
        i_p_one = indices_p[:, 0] # takes rows Xij=1
        i_m_one = indices_n[:, 0] #takes rows Xij=0
        i_zeros = indices_zeros[:, 0] #takes rows Xij=NA
        j_p_one = indices_p[:, 1] # cols Xij=1
        j_m_one = indices_n[:, 1] #cols Xij=0
        j_zeros = indices_zeros[:, 1] #cols Xij=NA

        ## Positives: Xij= 1 ### 
        xp = nu_a[i_p_one].flatten() + nu_p[:, j_p_one].flatten() #for A & C
        yp = nu_b[i_p_one].flatten() + nu_q[:, j_p_one].flatten() #for B & D

        sig_p = torch.sigmoid(mu_un+ nu_a[i_p_one]+ nu_b[i_p_one]+ nu_p[:, j_p_one].t()+ nu_q[:, j_p_one].t())
        der2_sig_p = (-sig_p * (1 - sig_p)).flatten()
        sum_var_p = (rho_a[i_p_one].flatten()+ rho_p[:, j_p_one].flatten()+ rho_b[i_p_one].flatten()+ rho_q[:, j_p_one].flatten())

        # f_{1}
        f = lambda x, y: torch.log(pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1))
        
        # Taylor development (1)
        expectation_taylor_p = (tau_1[i_p_one].view(-1, nq, 1)
            * tau_2[j_p_one].view(-1, 1, nl)
            * (f(xp, yp) + 0.5 * (der2_sig_p * sum_var_p).view(-1, 1, 1))).sum()

        ### Negatives: Xij= -1 ###
        xn = nu_a[i_m_one].flatten() + nu_p[:, j_m_one].flatten()
        yn = nu_b[i_m_one].flatten() + nu_q[:, j_m_one].flatten()
        
        sig_m = torch.sigmoid(mu_un+ nu_a[i_m_one]- nu_b[i_m_one]+ nu_p[:, j_m_one].t()- nu_q[:, j_m_one].t())
        der2_sig_m = -(sig_m * (1 - sig_m)).flatten()
        sum_var_m = (rho_a[i_m_one].flatten()+ rho_p[:, j_m_one].flatten()+ rho_b[i_m_one].flatten()+ rho_q[:, j_m_one].flatten())
        
        #f_{0}
        f = lambda x, y: torch.log((1 - pi).view(1, nq, nl)* torch.sigmoid(mu_un + x - y).view(-1, 1, 1))

        # Taylor development (0)
        expectation_taylor_m = (tau_1[i_m_one].view(-1, nq, 1)* tau_2[j_m_one].view(-1, 1, nl)
            * (f(xn, yn) + 0.5 * (der2_sig_m * sum_var_m).view(-1, 1, 1))).sum()

        ### Zeros: Xij= 0 ###

        # f_{NA}
        f = lambda x, y: torch.log(1- pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1)- (1 - pi.view(1, nq, nl))* torch.sigmoid(mu_un + x - y).view(-1, 1, 1))
        xz = nu_a[i_zeros].flatten() + nu_p[:, j_zeros].flatten()
        yz = nu_b[i_zeros].flatten() + nu_q[:, j_zeros].flatten()

        if device2:
            der_x = d2_DL3_XO(xz.view(-1, 1, 1).to(device2),yz.view(-1, 1, 1).to(device2),mu_un.to(device2),pi.view(1, nq, nl).to(device2),"x",).to(device)
        else:
            der_x = d2_DL3_XO(xz.view(-1, 1, 1),yz.view(-1, 1, 1),mu_un,pi.view(1, nq, nl),"x",)

        der_y = d2_DL3_XO(xz.view(-1, 1, 1),yz.view(-1, 1, 1),mu_un,pi.view(1, nq, nl),"y",)

        tau_12_ij = tau_1[i_zeros].view(-1, nq, 1) * tau_2[j_zeros].view(-1, 1, nl)
        
        # Taylor development (NA)
        expectation_taylor_zeros = (tau_12_ij* (f(xz, yz)+ 0.5
                * (der_x* (rho_a[i_zeros].flatten() + rho_p[:, j_zeros].flatten()).view(-1, 1, 1)
                    + der_y* (rho_b[i_zeros].flatten() + rho_q[:, j_zeros].flatten()).view(-1, 1, 1)))).sum()



        ### Final expectation ###
        expectation = (expectation_taylor_p+ expectation_taylor_m+ expectation_taylor_zeros)
        
        return (expectation if expectation < 0 else torch.tensor(np.inf, device=device))

def criteria(vector_of_parameters):
        (nu_a,rho_a,nu_b,rho_b,nu_p,rho_p,nu_q,rho_q,tau_1,tau_2,mu_un,sigma_sq_a,sigma_sq_b,sigma_sq_p,sigma_sq_q,alpha_1,alpha_2,
            pi) = reparametrized_expanded_params(vector_of_parameters,n1,n2,nq,nl,device,)        
        
        expectation = (entropy_rx(rho_a, rho_b, rho_p, rho_q, tau_1, tau_2)
            + expectation_loglike_A(nu_a, rho_a, sigma_sq_a)[0]
            + expectation_loglike_B(nu_b, rho_b, sigma_sq_b)[0]
            + expectation_loglike_P(nu_p, rho_p, sigma_sq_p)[0]
            + expectation_loglike_Q(nu_q, rho_q, sigma_sq_q)[0]
            + expectation_loglike_Y1(tau_1, alpha_1)
            + expectation_loglike_Y2(tau_2, alpha_2)
            + expectation_loglike_X_cond_ABPY1Y2(mu_un,nu_a,nu_b,nu_p,nu_q,rho_a,rho_b,rho_p,rho_q,tau_1,tau_2,pi)
        )
        return expectation