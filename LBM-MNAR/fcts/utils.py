import torch
from torch.nn.functional import softplus
import numpy as np
from scipy.special import expit, logit
import torch
import numpy as np
import yaml

def convert_to_yaml(obj):
    """
    Convert tensors, numpy arrays, and other objects to YAML-friendly types.

    Parameters:
    - obj: Object to be converted.

    Returns:
    - converted_obj: Converted object.

    Example:
    converted = convert_to_yaml_friendly(tensor_or_numpy_array)
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensor to list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    # Add more conversions for other types if needed
    else:
        return obj  # Return as is for other types

def save_objects_to_yaml(objects_dict, file_path):
    """
    Save objects (including tensors, numpy arrays, etc.) to a YAML file.

    Parameters:
    - objects_dict (dict): Dictionary containing objects to be saved.
    - file_path (str): Path to the YAML file.

    Example:
    save_objects_to_yaml({'tensor': torch.randn(3, 3), 'numpy_array': np.random.rand(4, 4)}, 'objects.yaml')
    """
    converted_dict = {key: convert_to_yaml(obj) for key, obj in objects_dict.items()}

    with open(file_path, 'w') as file:
        yaml.dump(converted_dict, file, default_flow_style=False)

def load_objects_from_yaml(file_path):
    """
    Load parameters from a YAML file.

    Parameters:
    - file_path (str): Path to the YAML file.

    Returns:
    - parameters (dict): Dictionary containing parameter names and values.

    Example:
    loaded_parameters = load_parameters_from_yaml('parameters.yaml')
    """
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters


inv_softplus=  lambda x: x + np.log(-np.expm1(-x))
shrink_simplex_internal= (lambda p: 1 - p[:, :-1] / np.cumsum(p[:, ::-1], axis=1)[:, :0:-1])
shrinkpow= lambda s: np.exp((np.arange(s.shape[1], 0, -1).reshape((1, -1))) * np.log(s))
shrink_simplex= lambda p: shrinkpow(shrink_simplex_internal(p))


def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = torch.autograd.grad(f.sum(), wrt, create_graph=True)[0]
        f = grads.sum()

    return grads


def expandpow(s, device):
    return torch.exp(
        (
            torch.ones(s.shape[1], device=device, dtype=torch.float32)
            / torch.arange(
                s.shape[1], 0, -1, device=device, dtype=torch.float32
            ).reshape((1, -1))
        )
        * torch.log(s)
    )


def expand_simplex_internal(x, device):
    return torch.cat(
        (
            torch.ones((x.shape[0], 1), device=device, dtype=torch.float32),
            torch.exp(torch.cumsum(torch.log(x), dim=1)),
        ),
        1,
    ) * torch.cat(
        (
            1 - x,
            torch.ones((x.shape[0], 1), device=device, dtype=torch.float32),
        ),
        1,
    )


def expand_simplex(x, device):
    return expand_simplex_internal(expandpow(x, device), device)


def split_params(gammatheta, n1, n2, nq, nl):
    assert len(gammatheta.shape) == 1
    assert (
        gammatheta.shape[0]
        == 4 * n1
        + 4 * n2
        + (n1 * (nq - 1))
        + (n2 * (nl - 1))
        + 5
        + nq
        - 1
        + nl
        - 1
        + nq * nl
    )

    n_gamma = (4 * n1 + 4 * n2 + (n1 * (nq - 1))) + (n2 * (nl - 1))

    r_nu_a = gammatheta[0:n1].reshape((n1, 1)) #a
    r_rho_a = gammatheta[n1 : (2 * n1)].reshape((n1, 1))

    r_nu_b = gammatheta[(2 * n1) : (3 * n1)].reshape((n1, 1)) #b
    r_rho_b = gammatheta[(3 * n1) : (4 * n1)].reshape((n1, 1))

    r_nu_p = gammatheta[(4 * n1) : (4 * n1 + n2)].reshape((1, n2)) #c
    r_rho_p = gammatheta[(4 * n1 + n2) : (4 * n1 + 2 * n2)].reshape((1, n2))

    r_nu_q = gammatheta[(4 * n1 + 2 * n2) : (4 * n1 + 3 * n2)].reshape((1, n2)) #d
    r_rho_q = gammatheta[(4 * n1 + 3 * n2) : (4 * n1 + 4 * n2)].reshape(
        (1, n2)
    )

    r_tau_1 = gammatheta[
        (4 * n1 + 4 * n2) : (4 * n1 + 4 * n2 + (n1 * (nq - 1)))
    ].reshape((n1, nq - 1))
    r_tau_2 = gammatheta[
        (4 * n1 + 4 * n2 + (n1 * (nq - 1))) : n_gamma
    ].reshape((n2, nl - 1))

    r_mu_un = gammatheta[n_gamma + 0].reshape((1, 1))

    r_sigma_sq_a = gammatheta[n_gamma + 1].reshape((1, 1))
    r_sigma_sq_b = gammatheta[n_gamma + 2].reshape((1, 1))
    r_sigma_sq_p = gammatheta[n_gamma + 3].reshape((1, 1))
    r_sigma_sq_q = gammatheta[n_gamma + 4].reshape((1, 1))

    r_alpha_1 = gammatheta[n_gamma + 5 : (n_gamma + 5 + nq - 1)].reshape(
        (nq - 1, 1)
    )
    r_alpha_2 = gammatheta[
        (n_gamma + 5 + nq - 1) : (n_gamma + 5 + nq - 1 + nl - 1)
    ].reshape((1, nl - 1))
    r_pi = gammatheta[
        (n_gamma + 5 + nq - 1 + nl - 1) : n_gamma
        + 5
        + nq
        - 1
        + nl
        - 1
        + nq * nl
    ].reshape((nq, nl))

    return (
        r_nu_a,
        r_rho_a,
        r_nu_b,
        r_rho_b,
        r_nu_p,
        r_rho_p,
        r_nu_q,
        r_rho_q,
        r_tau_1,
        r_tau_2,
        r_mu_un,
        r_sigma_sq_a,
        r_sigma_sq_b,
        r_sigma_sq_p,
        r_sigma_sq_q,
        r_alpha_1,
        r_alpha_2,
        r_pi,
        n1,
        n2,
        nq,
        nl,
    )


def reparametrized_expanded_params(gammatheta, n1, n2, nq, nl, device):
    (
        r_nu_a,
        r_rho_a,
        r_nu_b,
        r_rho_b,
        r_nu_p,
        r_rho_p,
        r_nu_q,
        r_rho_q,
        r_tau_1,
        r_tau_2,
        r_mu_un,
        r_sigma_sq_a,
        r_sigma_sq_b,
        r_sigma_sq_p,
        r_sigma_sq_q,
        r_alpha_1,
        r_alpha_2,
        r_pi,
        n1,
        n2,
        nq,
        nl,
    ) = split_params(gammatheta, n1, n2, nq, nl)
    nu_a = r_nu_a
    rho_a = softplus(r_rho_a)
    nu_b = r_nu_b
    rho_b = softplus(r_rho_b)
    nu_p = r_nu_p
    rho_p = softplus(r_rho_p)
    nu_q = r_nu_q
    rho_q = softplus(r_rho_q)
    tau_1 = expand_simplex(torch.sigmoid(r_tau_1), device)
    tau_2 = expand_simplex(torch.sigmoid(r_tau_2), device)
    mu_un = r_mu_un
    sigma_sq_a = softplus(r_sigma_sq_a)
    sigma_sq_b = softplus(r_sigma_sq_b)
    sigma_sq_p = softplus(r_sigma_sq_p)
    sigma_sq_q = softplus(r_sigma_sq_q)
    alpha_1 = expand_simplex(
        torch.sigmoid(r_alpha_1.reshape(1, nq - 1)), device
    ).reshape(nq, 1)
    alpha_2 = expand_simplex(
        torch.sigmoid(r_alpha_2.reshape(1, nl - 1)), device
    ).reshape(1, nl)
    pi = torch.sigmoid(r_pi)
    return (
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
    )


def d2_DL3_XO(x, y, mu, pi, d):
    sp = torch.sigmoid(y + x + mu)
    sm = torch.sigmoid(-y + x + mu)
    if d == "x":
        num = (
            pi ** 2 * sp ** 4
            - 2 * pi ** 2 * sm * sp ** 3
            + 2 * pi * sm * sp ** 3
            - pi ** 2 * sp ** 3
            - 2 * pi * sp ** 3
            + 2 * pi ** 2 * sm ** 2 * sp ** 2
            - 2 * pi * sm ** 2 * sp ** 2
            + pi ** 2 * sm * sp ** 2
            - pi * sm * sp ** 2
            + 3 * pi * sp ** 2
            - 2 * pi ** 2 * sm ** 3 * sp
            + 2 * pi * sm ** 3 * sp
            + pi ** 2 * sm ** 2 * sp
            - pi * sm ** 2 * sp
            - pi * sp
            + pi ** 2 * sm ** 4
            - 2 * pi * sm ** 4
            + sm ** 4
            - pi ** 2 * sm ** 3
            + 4 * pi * sm ** 3
            - 3 * sm ** 3
            - 3 * pi * sm ** 2
            + 3 * sm ** 2
            + pi * sm
            - sm
        )
    elif d == "y":
        num = (
            pi ** 2 * sp ** 4
            - 2 * pi ** 2 * sm * sp ** 3
            + 2 * pi * sm * sp ** 3
            - pi ** 2 * sp ** 3
            - 2 * pi * sp ** 3
            - 2 * pi ** 2 * sm ** 2 * sp ** 2
            + 2 * pi * sm ** 2 * sp ** 2
            + 5 * pi ** 2 * sm * sp ** 2
            - 5 * pi * sm * sp ** 2
            + 3 * pi * sp ** 2
            - 2 * pi ** 2 * sm ** 3 * sp
            + 2 * pi * sm ** 3 * sp
            + 5 * pi ** 2 * sm ** 2 * sp
            - 5 * pi * sm ** 2 * sp
            - 4 * pi ** 2 * sm * sp
            + 4 * pi * sm * sp
            - pi * sp
            + pi ** 2 * sm ** 4
            - 2 * pi * sm ** 4
            + sm ** 4
            - pi ** 2 * sm ** 3
            + 4 * pi * sm ** 3
            - 3 * sm ** 3
            - 3 * pi * sm ** 2
            + 3 * sm ** 2
            + pi * sm
            - sm
        )
    else:
        return None

    denum = (pi * sp - pi * sm + sm - 1) ** 2

    return num / denum


def init_random_params(n1, n2, nq, nl):
    """Function that generates gamma and theta for some intial parameters (clusters size, data matrix size)

    Args:
        n1 (int): row length
        n2 (int): column length
        nq (int): number of row clusters 
        nl (int): number of column clusters 

    Returns:
        np.concatenate((gamma, theta)): the respective parameters gamma and theta for the initial parameters
    """
    #Theta params
    mu_un = np.random.uniform(-4.5, -3.5) # expit(μ) is the global missingness rate
    sigma_sq_a = np.random.uniform(0.4, 0.7) 
    sigma_sq_b = np.random.uniform(0.4, 0.7)
    sigma_sq_p = np.random.uniform(0.4, 0.7)
    sigma_sq_q = np.random.uniform(0.4, 0.7)
    alpha_1 = (np.ones(nq) / nq).reshape((nq, 1)) #uniform proba of each row cluster
    alpha_2 = (np.ones(nl) / nl).reshape((1, nl)) #idem for col cluster
    pi = np.random.uniform(0.2, 0.8, (nq, nl)) #pi_kl (size: nb row clust  nb col clust)

    #Gamma params
    # non informative initialization 
    #### Moyennes tirées sur une uniforme [-0.5,0.5]
    nu_a = np.random.uniform(-0.5, 0.5, (n1, 1))# column vector of size n1
    nu_b = np.random.uniform(-0.5, 0.5, (n1, 1))
    nu_p = np.random.uniform(-0.5, 0.5, (1, n2)) # row vector of size n2
    nu_q = np.random.uniform(-0.5, 0.5, (1, n2))
    #### Variance fixe: 1e-5 pour tous 
    rho_a = 1e-5 * np.ones((n1, 1))# column vector of size n1
    rho_b = 1e-5 * np.ones((n1, 1))
    rho_p = 1e-5 * np.ones((1, n2)) # row vector of size n2
    rho_q = 1e-5 * np.ones((1, n2))
    tau_1 = np.diff(
        np.concatenate(
            (
                np.zeros((n1, 1)),
                np.sort(np.random.uniform(size=(n1, nq - 1)), axis=1),
                np.ones((n1, 1)),
            ),
            axis=1,
        ),
        axis=1,
    ) #size (n1,nq)
    tau_2 = np.diff(
        np.concatenate(
            (
                np.zeros((n2, 1)),
                np.sort(np.random.uniform(size=(n2, nl - 1)), axis=1),
                np.ones((n2, 1)),
            ),
            axis=1,
        ),
        axis=1,
    ) #size (n2, nl)
    theta = np.concatenate(
        (
            (mu_un,),
            (inv_softplus(sigma_sq_a),),
            (inv_softplus(sigma_sq_b),),
            (inv_softplus(sigma_sq_p),),
            (inv_softplus(sigma_sq_q),),
            logit(shrink_simplex(alpha_1.T).flatten()),
            logit(shrink_simplex(alpha_2).flatten()),
            logit(pi.flatten()),
        )
    )
    gamma = np.concatenate(
        (
            nu_a.flatten(),
            inv_softplus(rho_a.flatten()),
            nu_b.flatten(),
            inv_softplus(rho_b.flatten()),
            nu_p.flatten(),
            inv_softplus(rho_p.flatten()),
            nu_q.flatten(),
            inv_softplus(rho_q.flatten()),
            logit(shrink_simplex(tau_1).flatten()),
            logit(shrink_simplex(tau_2).flatten()),
        )
    )
    assert len(theta.shape) == 1
    assert theta.shape[0] == 5 + nq - 1 + nl - 1 + nq * nl
    assert len(gamma.shape) == 1
    assert gamma.shape[0] == 4 * n1 + 4 * n2 + (n1 * (nq - 1)) + (
        n2 * (nl - 1)
    )

    return np.concatenate((gamma, theta))
