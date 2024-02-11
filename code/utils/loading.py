import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def parse_spec(path: str) -> Tuple[int, str, torch.Tensor, float]:
    """Returns label, dataset, image and epsilon from a spec file

    Args:
        path (str): Path to spec file

    Returns:
        Tuple[int, str, torch.Tensor, float]: Label, image and epsilon
    """

    # Get epsilon from filename
    eps = float(".".join(path.split("/")[-1].split("_")[-1].split(".")[:2]))
    # Get dataset from filename
    dataset = path.split("/")[-1].split("_")[1]

    shape = (1, 28, 28) if "mnist" in dataset else (3, 32, 32)

    with open(path, "r") as f:
        # First line is the label
        label = int(f.readline().strip())
        # Second line is the image
        image = [float(x) for x in f.readline().strip().split(",")]

    return label, dataset, torch.tensor(image).reshape(shape), eps




def concretize_bounds(cu: torch.Tensor, 
                      cu_b: torch.Tensor,
                      cl: torch.Tensor,
                      cl_b: torch.Tensor,
                      lb: torch.Tensor,
                      ub: torch.Tensor):
    """
    Returns concrete bounds for the provided set of abstract constraints. 
    lb and ub contain the concrete bounds for the variables present in the symbolic constraints.

    Args:
        - cu: symbolic upper bound constraints (n_constraints, n_vars)
        - cu_b: symbolic upper bound constraints bias (n_constraints)
        - cl: symbolic lower bound constraints (n_constraints, n_vars)
        - cl_b: symbolic lower bound constraints bias (n_constraints)
        - lb: concrete lower bound (n_vars) 
        - ub: concrete upper bound (n_vars)


    returns:
        - cu_conc: concrete upper bound constraints (n_constraints)
        - cl_conc: concrete lower bound constraints (n_constraints)
    """
    assert cu.shape[-2] == cu_b.shape[-1] , f"cu shape: {cu.shape} cu_b shape: {cu_b.shape}"
    assert cl.shape[-2] == cl_b.shape[-1], f"cl shape: {cl.shape} cl_b shape: {cl_b.shape}"
    assert cu.shape[-1] == cl.shape[-1], f"cu shape: {cu.shape} cl shape: {cl.shape}"
    assert cu.shape[-1] == ub.shape[-1], f"cu shape: {cu.shape} ub shape: {ub.shape}"
    assert cl.shape[-1] == lb.shape[-1], f"cl shape: {cl.shape} lb shape: {lb.shape}"

    #Get the masks on the the symbolic constraints.
    #If the coefficient is positive then we fetch the corresponding bound (concrete upper for symbolic upper and concrete lower for symbolic lower)
    #Else we fetch the opposite bound (concrete lower for symbolic upper and concrete upper for symbolic lower)
    mask_upper = torch.where(cu > 0, torch.tensor(1.0), torch.tensor(0.0))

    mask_lower = torch.where(cl > 0, torch.tensor(1.0), torch.tensor(0.0))




    lb = torch.transpose(lb.unsqueeze(-1).repeat_interleave(cl.shape[0], dim=-1), dim0=-1, dim1=-2)
    ub = torch.transpose(ub.unsqueeze(-1).repeat_interleave(cu.shape[0], dim=-1), dim0=-1, dim1=-2)


    #Compute the corresponding concrete bounds for the symbolic constraints

    ub_temp = mask_upper * ub + (1 - mask_upper) * lb
    lb_temp = mask_lower * lb + (1 - mask_lower) * ub

  


    #Compute the concrete bounds for the symbolic constraints
    #Element wise multiplication followed by a sum over the last dimension


    cu_conc = cu * ub_temp
    cl_conc = cl * lb_temp



    cu_conc = torch.sum(cu_conc, dim=-1)
    cl_conc = torch.sum(cl_conc, dim=-1)


    #Suming over the last dimension to obtain the innerproduct
    cu_conc = cu_conc + cu_b
    cl_conc = cl_conc + cl_b

    assert (cu_conc >= cl_conc).all(), "concretize_bounds - Upper bound is smaller than lower bound"

    return cu_conc, cl_conc



def backsub_relu(cu: torch.Tensor, 
                cu_b: torch.Tensor,
                cl: torch.Tensor,
                cl_b: torch.Tensor,
                lambda_u: torch.Tensor,
                mu_u: torch.Tensor,
                lambda_l: torch.Tensor,
                mu_l: torch.Tensor):
    """
    Returns the symbolic constraints obtained after backsubstituting the relu constraints into a system of linear constraints
    
    
    Arguments:
        - cu: symbolic upper bound constraints (n_constraints, n_vars)
        - cu_b: symbolic upper bound constraints bias (n_constraints)
        - cl: symbolic lower bound constraints (n_constraints, n_vars)
        - cl_b: symbolic lower bound constraints bias (n_constraints)
        - lambda_u: symbolic upper bound relu constraints (n_vars)
        - mu_u: symbolic upper bound relu constraints bias (n_vars)
        - lambda_l: symbolic lower bound relu constraints (n_vars)
        - mu_l: symbolic lower bound relu constraints bias (n_var)


    returns: 

        Note that altough the dimension of the vars is the same, these refer to different variables.

        xi = max(0, xj) = relu(xj) Our system was defined as function of the xi variables now it is defined as a function of the xj variables

        - cu_new: symbolic upper bound constraints after backsubstitution through the relu (n_constraints, n_vars)
        - cu_b_new: symbolic upper bound constraints bias after backsubstitution through the relu (n_constraints)
        - cl_new: symbolic lower bound constraints after backsubstitution through the relu (n_constraints, n_vars)
        - cl_b_new: symbolic lower bound constraints bias after backsubstitution through the relu (n_constraints)
    """


    assert cu.shape[-2] == cu_b.shape[-1], f"cu shape: {cu.shape} cu_b shape: {cu_b.shape}"
    assert cl.shape[-2] == cl_b.shape[-1], f"cl shape: {cl.shape} cl_b shape: {cl_b.shape}"
    assert lambda_u.shape[-1] == mu_u.shape[-1], f"lambda_u shape: {lambda_u.shape} mu_u shape: {mu_u.shape}"
    assert lambda_l.shape[-1] == mu_l.shape[-1], f"lambda_l shape: {lambda_l.shape} mu_l shape: {mu_l.shape}"
    assert cu.shape[-2] == cl.shape[-2], f"cu shape: {cu.shape} cl shape: {cl.shape}"
    assert cu.shape[-1] == lambda_u.shape[-1], f"cu shape: {cu.shape} lambda_u shape: {lambda_u.shape}"



    mask_upper = torch.where(cu > 0, torch.tensor(1.0), torch.tensor(0.0))
    mask_lower = torch.where(cl > 0, torch.tensor(1.0), torch.tensor(0.0))




    #We duplicate along in the following way:
    """
    lambda_u = [[lambda_u_1, lambda_u_2, ..., lambda_u_n_vars],
                [lambda_u_1, lambda_u_2, ..., lambda_u_n_vars],
                ...
                [lambda_u_1, lambda_u_2, ..., lambda_u_n_vars]]
    
    """


    

    lambda_l = torch.transpose(lambda_l.unsqueeze(-1), dim0=-1, dim1=-2).repeat(cl.shape[-2],  1)

    lambda_u = torch.transpose(lambda_u.unsqueeze(-1), dim0=-1, dim1=-2).repeat(cu.shape[-2],  1)

    mu_u = torch.transpose(mu_u.unsqueeze(-1), dim0=-1, dim1=-2).repeat(cu.shape[-2],  1)
    mu_l = torch.transpose(mu_l.unsqueeze(-1), dim0=-1, dim1=-2).repeat(cu.shape[-2],  1)

    #We preserve the same structure of matrix but for every entry we select the corresponding value 
    #from the upper or lower bound depending on the sign of the coefficient for that variable in that constraint
    masked_lambda_u = mask_upper * lambda_u + (1 - mask_upper) * lambda_l
    masked_mu_u = mask_upper * mu_u + (1 - mask_upper) * mu_l
    masked_lambda_l = mask_lower * lambda_l + (1 - mask_lower) * lambda_u
    masked_mu_l = mask_lower * mu_l + (1 - mask_lower) * mu_u

    cu_new = cu * masked_lambda_u
    cl_new = cl * masked_lambda_l

    cu_b_new = cu_b + torch.sum(cu * masked_mu_u, dim=-1)
    cl_b_new = cl_b + torch.sum(cl * masked_mu_l, dim=-1)

    return cu_new, cu_b_new, cl_new, cl_b_new



