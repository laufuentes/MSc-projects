import torch
import numpy as np
from torch import nn
from fcts.utils import reparametrized_expanded_params, d2_DL3_XO


class LBM_NMAR(nn.Module):
    def __init__(self, init_parameters, votes, shapes, device, device2=None):
        super().__init__()
        self.indices_p = np.argwhere(votes == 1) #argwhere: matrix with couples (row,column) with 1 values 
        self.indices_n = np.argwhere(votes == -1) #idem with -1
        self.indices_zeros = np.argwhere(votes == 0) #idem with 0 

        self.n1 = shapes[0]
        self.n2 = shapes[1]
        self.nq = shapes[2]
        self.nl = shapes[3]
        self.device = device
        self.device2 = device2
        lengamma = (
            4 * self.n1
            + 4 * self.n2
            + (self.n1 * (self.nq - 1))
            + (self.n2 * (self.nl - 1))
        )
        self.variationnal_params = nn.Parameter(
            init_parameters[:lengamma].clone()
        )
        self.model_params = nn.Parameter(init_parameters[lengamma:].clone())

    def forward(self, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.criteria()
        else:
            return self.criteria()
    def criteria(self):
        (
            nu_a,
            rho_a,
            nu_b,
            rho_b,
            nu_p,
            rho_p,
            nu_q,
            rho_q,
            tau_1,
            tau_2,
            mu_un,
            sigma_sq_a,
            sigma_sq_b,
            sigma_sq_p,
            sigma_sq_q,
            alpha_1,
            alpha_2,
            pi,
        ) = reparametrized_expanded_params(
            torch.cat((self.variationnal_params, self.model_params)),
            self.n1,
            self.n2,
            self.nq,
            self.nl,
            self.device,
        )
        if torch.any(tau_1.sum(dim=0) < 0.5):
            print("One empty row class, algo stoped")
            return torch.tensor(np.nan, device=self.device)
        if torch.any(tau_2.sum(dim=0) < 0.5):
            print("One empty column class, algo stoped")
            return torch.tensor(np.nan, device=self.device)

        expectation = (
            self.entropy_rx(rho_a, rho_b, rho_p, rho_q, tau_1, tau_2)
            + self.expectation_loglike_A(nu_a, rho_a, sigma_sq_a)[0]
            + self.expectation_loglike_B(nu_b, rho_b, sigma_sq_b)[0]
            + self.expectation_loglike_P(nu_p, rho_p, sigma_sq_p)[0]
            + self.expectation_loglike_Q(nu_q, rho_q, sigma_sq_q)[0]
            + self.expectation_loglike_Y1(tau_1, alpha_1)
            + self.expectation_loglike_Y2(tau_2, alpha_2)
            + self.expectation_loglike_X_cond_ABPY1Y2(
                mu_un,
                nu_a,
                nu_b,
                nu_p,
                nu_q,
                rho_a,
                rho_b,
                rho_p,
                rho_q,
                tau_1,
                tau_2,
                pi,
            )
        )

        return -expectation

    def entropy_rx(self, rho_a, rho_b, rho_p, rho_q, tau_1, tau_2):
        n1, n2 = self.n1, self.n2
        return (
            1/2* (2 * n1 + 2 * n2)* 
            (torch.log(torch.tensor(2 * np.pi, dtype=torch.float32, device=self.device))+ 1)
            + 1 / 2 * torch.sum(torch.log(rho_a))
            + 1 / 2 * torch.sum(torch.log(rho_b))
            + 1 / 2 * torch.sum(torch.log(rho_p))
            + 1 / 2 * torch.sum(torch.log(rho_q))
            - torch.sum(tau_1 * torch.log(tau_1))
            - torch.sum(tau_2 * torch.log(tau_2))
        )

    def expectation_loglike_A(self, nu_a, rho_a, sigma_sq_a):
        n1 = self.n1
        return -n1 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.float32, device=self.device)
            )
            + torch.log(sigma_sq_a)
        ) - 1 / (2 * sigma_sq_a) * torch.sum(rho_a + nu_a ** 2)

    def expectation_loglike_B(self, nu_b, rho_b, sigma_sq_b):
        n1 = self.n1
        return -n1 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.float32, device=self.device)
            )
            + torch.log(sigma_sq_b)
        ) - 1 / (2 * sigma_sq_b) * torch.sum(rho_b + nu_b ** 2)

    def expectation_loglike_P(self, nu_p, rho_p, sigma_sq_p):
        n2 = self.n2
        return -n2 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.float32, device=self.device)
            )
            + torch.log(sigma_sq_p)
        ) - 1 / (2 * sigma_sq_p) * torch.sum(rho_p + nu_p ** 2)

    def expectation_loglike_Q(self, nu_q, rho_q, sigma_sq_q):
        n2 = self.n2
        return -n2 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.float32, device=self.device)
            )
            + torch.log(sigma_sq_q)
        ) - 1 / (2 * sigma_sq_q) * torch.sum(rho_q + nu_q ** 2)

    def expectation_loglike_Y1(self, tau_1, alpha_1):
        n1, nq = self.n1, self.nq
        return tau_1.sum(0) @ torch.log(alpha_1)

    def expectation_loglike_Y2(self, tau_2, alpha_2):
        n2, nl = self.n2, self.nl
        return tau_2.sum(0) @ torch.log(alpha_2).t()

    def expectation_loglike_X_cond_ABPY1Y2(
        self,
        mu_un,
        nu_a,
        nu_b,
        nu_p,
        nu_q,
        rho_a,
        rho_b,
        rho_p,
        rho_q,
        tau_1,
        tau_2,
        pi,
    ):
        n1, n2, nq, nl = self.n1, self.n2, self.nq, self.nl
        indices_p, indices_n, indices_zeros = (
            self.indices_p,
            self.indices_n,
            self.indices_zeros,
        )
        i_p_one = indices_p[:, 0] # takes rows Xij=1
        i_m_one = indices_n[:, 0] #takes rows Xij=0
        i_zeros = indices_zeros[:, 0] #takes rows Xij=NA
        j_p_one = indices_p[:, 1] # cols Xij=1
        j_m_one = indices_n[:, 1] #cols Xij=0
        j_zeros = indices_zeros[:, 1] #cols Xij=NA

        ## POSITIVES (Xij=1) ### 
        xp = nu_a[i_p_one].flatten() + nu_p[:, j_p_one].flatten() #for A & C
        yp = nu_b[i_p_one].flatten() + nu_q[:, j_p_one].flatten() #for B & D
        sig_p = torch.sigmoid(
            mu_un
            + nu_a[i_p_one]
            + nu_b[i_p_one]
            + nu_p[:, j_p_one].t()
            + nu_q[:, j_p_one].t()
        )
        der2_sig_p = (-sig_p * (1 - sig_p)).flatten()
        sum_var_p = (
            rho_a[i_p_one].flatten()
            + rho_p[:, j_p_one].flatten()
            + rho_b[i_p_one].flatten()
            + rho_q[:, j_p_one].flatten()
        )

        f = lambda x, y: torch.log(
            pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1)
        )
        expectation_taylor_p = (
            tau_1[i_p_one].view(-1, nq, 1)
            * tau_2[j_p_one].view(-1, 1, nl)
            * (f(xp, yp) + 0.5 * (der2_sig_p * sum_var_p).view(-1, 1, 1))
        ).sum()

        ### NEGATIVES ###
        xn = nu_a[i_m_one].flatten() + nu_p[:, j_m_one].flatten()
        yn = nu_b[i_m_one].flatten() + nu_q[:, j_m_one].flatten()
        sig_m = torch.sigmoid(
            mu_un
            + nu_a[i_m_one]
            - nu_b[i_m_one]
            + nu_p[:, j_m_one].t()
            - nu_q[:, j_m_one].t()
        )
        der2_sig_m = -(sig_m * (1 - sig_m)).flatten()
        sum_var_m = (
            rho_a[i_m_one].flatten()
            + rho_p[:, j_m_one].flatten()
            + rho_b[i_m_one].flatten()
            + rho_q[:, j_m_one].flatten()
        )
        f = lambda x, y: torch.log(
            (1 - pi).view(1, nq, nl)
            * torch.sigmoid(mu_un + x - y).view(-1, 1, 1)
        )

        expectation_taylor_m = (
            tau_1[i_m_one].view(-1, nq, 1)
            * tau_2[j_m_one].view(-1, 1, nl)
            * (f(xn, yn) + 0.5 * (der2_sig_m * sum_var_m).view(-1, 1, 1))
        ).sum()

        ### ZEROS ###

        f = lambda x, y: torch.log(
            1
            - pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1)
            - (1 - pi.view(1, nq, nl))
            * torch.sigmoid(mu_un + x - y).view(-1, 1, 1)
        )
        xz = nu_a[i_zeros].flatten() + nu_p[:, j_zeros].flatten()
        yz = nu_b[i_zeros].flatten() + nu_q[:, j_zeros].flatten()

        if self.device2:
            der_x = d2_DL3_XO(
                xz.view(-1, 1, 1).to(self.device2),
                yz.view(-1, 1, 1).to(self.device2),
                mu_un.to(self.device2),
                pi.view(1, nq, nl).to(self.device2),
                "x",
            ).to(self.device)
        else:
            der_x = d2_DL3_XO(
                xz.view(-1, 1, 1),
                yz.view(-1, 1, 1),
                mu_un,
                pi.view(1, nq, nl),
                "x",
            )
        der_y = d2_DL3_XO(
            xz.view(-1, 1, 1),
            yz.view(-1, 1, 1),
            mu_un,
            pi.view(1, nq, nl),
            "y",
        )
        tau_12_ij = tau_1[i_zeros].view(-1, nq, 1) * tau_2[j_zeros].view(
            -1, 1, nl
        )
        expectation_taylor_zeros = (
            tau_12_ij
            * (
                f(xz, yz)
                + 0.5
                * (
                    der_x
                    * (
                        rho_a[i_zeros].flatten() + rho_p[:, j_zeros].flatten()
                    ).view(-1, 1, 1)
                    + der_y
                    * (
                        rho_b[i_zeros].flatten() + rho_q[:, j_zeros].flatten()
                    ).view(-1, 1, 1)
                )
            )
        ).sum()

        expectation = (
            expectation_taylor_p
            + expectation_taylor_m
            + expectation_taylor_zeros
        )
        return (
            expectation
            if expectation < 0
            else torch.tensor(np.inf, device=self.device)
        )
