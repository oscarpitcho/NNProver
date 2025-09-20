import torch
import torch.nn.functional as F
from utils.loading import concretize_bounds, backsub_relu
from math import sqrt
import numpy as np
import itertools
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

class DeepPoly:
    def __init__(self, net : torch.nn.Sequential, input_size : int):
        """
        Neural network verifier using the DeepPoly abstract domain for computing sound output bounds.
        
        DeepPoly provides formal verification of neural network properties by propagating symbolic
        constraints and concrete bounds through the network layers. It computes guaranteed 
        over-approximations of network outputs given box-constrained inputs.
        
        The verifier transforms each layer into an abstract transformer that maintains both:
        - Symbolic linear constraints (expressing outputs as functions of inputs)
        - Concrete interval bounds (min/max values for each neuron)
        
        For non-linear layers (ReLU), it uses convex relaxations with learnable parameters
        to obtain tighter bounds through backsubstitution.
        
        Parameters
        ----------
        net : torch.nn.Sequential
            The neural network to verify. Must be a sequential model containing
            supported layers (Linear, Conv2d, ReLU, LeakyReLU).
        input_size : int
            The total size of the flattened input to the network.
            For images: input_size = channels * height * width.
        """
        super(DeepPoly, self).__init__()
        # These variables will store the system of the output layer over time. 
        # There will always be n_classes == n_constraints and the number of variables will change each time we backsub
        self.net = net
        self.last = None

        self.input_box_bounds_upper = None
        self.input_box_bounds_lower = None
        
       #Create the abstract network with the transformers of each layer
        prevl_input_size = input_size
        verifier_net = []
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                self.last = LinearTransformer(layer)
                verifier_net.append(self.last)
                prevl_input_size = layer.out_features

            elif isinstance(layer, torch.nn.Conv2d):
                self.last = ConvTransformer(layer, prevl_input_size)
                verifier_net.append(self.last)
                prevl_input_size = self.last.out_features

            elif isinstance(layer, torch.nn.ReLU):
                self.last = ReLuTransformer(layer, prevl_input_size)
                verifier_net.append(self.last)

            elif isinstance(layer, torch.nn.LeakyReLU):
                self.last = LeakyReLuTransformer(layer, prevl_input_size)
                verifier_net.append(self.last)
        
        self.verifier_net = torch.nn.Sequential(*verifier_net)
        
    #To maintain the same interface
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def backsubstitute_from_layer(self, from_layer : int):
        """
        uc, uc_b, lc, lc_b contain the upper and lower symbolic constraints for the layer from_layer.

        They have one row / constraint per neuron in the layer from_layer. We are going to refine these constraints via backsubstitution
        to thenn compute more precise concrete bounds on the of from layer
        """

        uc = self.verifier_net[from_layer].uc
        uc_b = self.verifier_net[from_layer].uc_b
        lc = self.verifier_net[from_layer].lc
        lc_b = self.verifier_net[from_layer].lc_b

        #
        for i in range(from_layer - 1 , -1, -1):
            uc, uc_b, lc, lc_b = self.verifier_net[i].backwards(uc, uc_b, lc, lc_b)

        #Concretize the new bounds using the function
        upper_bounds, lower_bounds = concretize_bounds(uc, uc_b, lc, lc_b, self.input_box_bounds_lower, self.input_box_bounds_upper)
        return lower_bounds, upper_bounds

    def forward(self, lower_bounds : torch.Tensor, upper_bounds : torch.Tensor):
        """
        Arguments:
        - The box bounds over the input domain which will be tested
            - lower_bounds (n_batch, n_channels, height, width) or (n_channels, height, width) if there are no batches
            - upper_bounds (n_batch, n_channels, height, width) or (n_channels, height, width) if there are no batches
            
        Returns: 
        - The concrete bounds over the prediction neurons of the network
            - lower_bounds (n_batch, n_classes) or (n_classes) if there are no batches
            - upper_bounds (n_batch, n_classes) or (n_classes) if there are no batches
        """
        lower_bounds = lower_bounds.reshape(-1).clone()
        upper_bounds = upper_bounds.reshape(-1).clone() 
        
        self.input_box_bounds_upper = upper_bounds
        self.input_box_bounds_lower = lower_bounds

        #Propagating updated bounds through network with backsub
        for i, layer in enumerate(self.verifier_net):
            logger.debug(f"Deeppoly verifier - Calling Forward pass layer {i} - {layer._get_name()}")
            if isinstance(layer, LinearTransformer):
               lower_bounds, upper_bounds = layer(lower_bounds, upper_bounds)
            if isinstance(layer, ConvTransformer):
               lower_bounds, upper_bounds = layer(lower_bounds, upper_bounds)

            #In the case of Relu and LeakyRelu we always backsub before. 
            #By obtaining tighter bounds it might show that we can 
            #do exact bounds l,u <0 or 0<l,u and thus avoid approximation.
            if isinstance(layer, ReLuTransformer):
               lower_bounds, upper_bounds = self.backsubstitute_from_layer(i-1) #
               lower_bounds, upper_bounds = layer(lower_bounds, upper_bounds)
            if isinstance(layer, LeakyReLuTransformer):
               lower_bounds, upper_bounds = self.backsubstitute_from_layer(i-1)
               lower_bounds, upper_bounds = layer(lower_bounds, upper_bounds)
            assert (upper_bounds >= lower_bounds).all(), f"Upper bound is smaller than lower bound at layer {i+1}/{len(self.verifier_net)}: {layer}"
            
        logger.debug(f"Forward pass done - Verifier net  : {self.verifier_net}")
        logger.debug(f"-------Concrete bounds state-----------")
        for i, layer in enumerate(self.verifier_net):
            logger.debug(f"Concrete bounds for layer first 5 valus {i} - {layer} - lb: {layer.lb[:5]}, ub: {layer.ub[:5]}")

        #For maximum precision we backsubstitute through the entire network.
        lower_bounds, upper_bounds = self.backsubstitute_from_layer(len(self.verifier_net) - 1)
        return lower_bounds, upper_bounds
    

class LinearTransformer(torch.nn.Module):
    def __init__(self, layer : torch.nn.Linear):
        super(LinearTransformer, self).__init__()
        self.layer = layer
        """
        These are the constraints in shape (size_layer_out, size_layer_in)
        i.e alternatively n_constraints, n_vars
        """
        self.uc = layer.weight
        self.uc_b = layer.bias
        self.lc = layer.weight
        self.lc_b = layer.bias
    
        self.ub = torch.zeros(layer.out_features)
        self.lb = torch.zeros(layer.out_features)
        
    def forward(self, lower_bounds : torch.Tensor, upper_bounds : torch.Tensor):
        #python code/verifier.py --net fc_base --spec test_cases/fc_base/img0_mnist_0.2456.txt --> not verified basic
        #python code/verifier.py --net fc_base --spec test_cases/fc_base/img2_mnist_0.0784.txt --> verified basic

        logger.info(f'Linear layer propagation... {self.layer}')
        self.lb = lower_bounds.clone()
        self.ub = upper_bounds.clone()
        self.ub, self.lb = concretize_bounds(self.uc, self.uc_b, self.lc, self.lc_b, self.lb, self.ub)
        
        logger.debug(f"linear_forward - cl_conc_out first 5 values {self.lb[:5]}")
        logger.debug(f"linear_forward - cu_conc_out first 5 values {self.ub[:5]}")

        #print(f"UPPER CONSTRAINTS shape: {self.uc.shape}")
        return self.lb, self.ub

    @staticmethod
    def do_backsub(uc : torch.Tensor, uc_b : torch.Tensor, 
                   lc : torch.Tensor, lc_b: torch.Tensor, 
                   layer_constraints : torch.Tensor, layer_constraints_b : torch.Tensor):
            """Used for testing and modularity puposes"""
            new_uc = torch.matmul(uc, layer_constraints) 
            new_uc_b = torch.matmul(uc, layer_constraints_b) + uc_b

            new_lc = torch.matmul(lc, layer_constraints)
            new_lc_b = torch.matmul(lc, layer_constraints_b) + lc_b
            return new_uc, new_uc_b, new_lc, new_lc_b

    def backwards(self, uc: torch.Tensor, uc_b: torch.Tensor, lc: torch.Tensor, lc_b: torch.Tensor):
        """
        Applies the backsubstitution of the layer to the provided symbolic constraints.

        It assumes that the Transformer has been backsubstituted up to the next layer (i), in other words, 
        the symbolic constraints are defined using the variables of layer i and the concrete bounds
        are computed by feeding the bounds of layer i - 1 in. 

        Args:
            - uc: symbolic upper bound constraints (n_constraints_last_layer, n_vars_current_layer)
            - uc_b: symbolic upper bound constraints bias (n_constraints_last_layer)
            - lc: symbolic lower bound constraints (n_constraints_last_layer, n_vars_current_layer)
            - lc_b: symbolic lower bound constraints bias (n_constraints_last_layer)
            
        Returns:
            - uc: symbolic upper bound constraints for previous layer (n_constraints_last_layer, n_vars_previous_layer)
            - uc_b: symbolic upper bound constraints bias for previous layer (n_constraints_last_layer)
            - lc: symbolic lower bound constraints for previous layer (n_constraints_last_layer, n_vars_previous_layer)
            - lc_b: symbolic lower bound constraints bias for previous layer (n_constraints_last_layer)     
        """        
        return self.do_backsub(uc, uc_b, lc, lc_b, self.uc, self.uc_b)
    
class ConvTransformer(torch.nn.Module):
    def __init__(self, layer : torch.nn.Conv2d, input_size : int):
        super(ConvTransformer, self).__init__()
        #Parameters of the convolutional layer
        self.layer = layer
        self.weight = self.layer.weight.detach() 
        self.bias = self.layer.bias.detach()

        self.padding = self.layer.padding[0]
        self.stride = self.layer.stride[0]

        self.filter_height = self.layer.kernel_size[0]
        self.filter_width = self.layer.kernel_size[1]

        # Compute the dimensions of the output feature map
        self.input_height = int(sqrt(input_size // self.layer.in_channels))
        self.input_width = int(sqrt(input_size // self.layer.in_channels))

        self.out_height = (self.input_height - self.filter_height + 2 * self.padding) // self.stride + 1
        self.out_width = (self.input_width - self.filter_width + 2 * self.padding) // self.stride + 1

        self.out_features = self.layer.out_channels * self.out_height * self.out_width
        self.n_constraints = self.out_features

        # Build the weight matrix that 
        weight_matrix = self.build_weight_matrix()
        self.constraints = weight_matrix

        # Adjusting the shapes of bias tensors
        if self.layer.bias is not None:
            self.constraints_b = self.layer.bias.view(1, -1, 1, 1).repeat(1, 1, self.out_height, self.out_width).view(self.n_constraints).clone()
        else:
            self.constraints_b = torch.zeros(self.n_constraints)
        
        self.uc = weight_matrix
        self.uc_b = self.constraints_b
        self.lc = weight_matrix
        self.lc_b = self.constraints_b

        self.ub = torch.zeros(self.n_constraints)
        self.lb = torch.zeros(self.n_constraints)

        assert self.constraints.shape == (self.n_constraints, input_size)
        assert self.constraints_b.shape == (self.n_constraints,)
    
    def build_weight_matrix(self):
        """Transforms a convolutional layer into an equivalent linear layer."""

        width_with_padding = self.input_width + 2 * self.padding
        height_with_padding = self.input_height + 2 * self.padding
        spatial_dim_with_padding = height_with_padding * width_with_padding

        # Initialize result matrix
        result_matrix = torch.zeros((self.out_height * self.out_width * self.layer.out_channels,
                                        spatial_dim_with_padding * self.layer.in_channels))
        
        # Construct row fillers
        filler_length = (self.layer.in_channels - 1) * spatial_dim_with_padding + \
                        (self.filter_height - 1) * width_with_padding + self.filter_height

        row_fillers = torch.zeros((self.layer.out_channels, filler_length))
        
        # Fill the row fillers with the weights of the convolutional layer
        for out_channel, in_channel, kernel_row in itertools.product(range(self.layer.out_channels), 
                                                                    range(self.layer.in_channels), 
                                                                    range(self.filter_height)):
            start_index = in_channel * spatial_dim_with_padding + kernel_row * width_with_padding
            row_fillers[out_channel, start_index: start_index + self.filter_height] =\
                self.layer.weight[out_channel, in_channel, kernel_row]

        # Populate the result matrix with the row fillers 
        for out_channel, out_height, out_width in itertools.product(range(self.layer.out_channels), 
                                                                    range(self.out_height), 
                                                                    range(self.out_width)):
            row_offset = out_height * self.stride * width_with_padding + out_width * self.stride
            neuron_index = out_channel * self.out_height * self.out_width +\
                        out_height * self.out_width + out_width
            result_matrix[neuron_index, row_offset: row_offset + filler_length] = row_fillers[out_channel]

        # Remove padding columns from the result matrix 
        cols_to_remove = [in_channel * spatial_dim_with_padding + in_height * width_with_padding + in_width
                        for in_channel, in_height, in_width in itertools.product(range(self.layer.in_channels), 
                                                                                range(height_with_padding), 
                                                                                range(width_with_padding))
                        if in_width < self.padding or in_width >= self.padding + self.input_width or
                        in_height < self.padding or in_height >= self.padding + self.input_height]

        cols_to_remove = list(np.unique(np.array(cols_to_remove)))
        result_matrix = torch.from_numpy(np.delete(result_matrix.detach().numpy(), cols_to_remove, axis=1))

        return result_matrix
        
    def forward(self, lower_bounds : torch.Tensor, upper_bounds : torch.Tensor):
        #python code/verifier.py --net conv_base --spec test_cases/conv_base/img0_mnist_0.0707.txt
        #python code/verifier.py --net conv_base --spec test_cases/conv_base/img1_mnist_0.0014.txt
        #python code/verifier.py --net conv_base --spec test_cases/conv_base/img2_mnist_0.0494.txt
        logger.info(f'Convolutional layer propagation...{self.layer}')
        
        self.lb = lower_bounds.clone()
        self.ub = upper_bounds.clone()
       
        self.ub,self.lb = concretize_bounds(self.constraints, self.constraints_b, self.constraints, self.constraints_b, self.lb, self.ub)
        return self.lb, self.ub
    
    @staticmethod
    def do_backsub(uc : torch.Tensor, uc_b : torch.Tensor, 
                   lc : torch.Tensor, lc_b: torch.Tensor, 
                   layer_contstraints : torch.Tensor, layer_constraints_b : torch.Tensor):
            """Used for testing puposes"""
            new_uc = torch.matmul(uc, layer_contstraints) 
            new_uc_b = torch.matmul(uc, layer_constraints_b) + uc_b

            new_lc = torch.matmul(lc, layer_contstraints)
            new_lc_b = torch.matmul(lc, layer_constraints_b) + lc_b
            return new_uc, new_uc_b, new_lc, new_lc_b
    
    def backwards(self, uc: torch.Tensor, uc_b: torch.Tensor,  lc : torch.Tensor, lc_b : torch.Tensor):
        return self.do_backsub(uc, uc_b, lc, lc_b, self.uc, self.uc_b)
            
class ReLuTransformer(torch.nn.Module):
    def __init__(self, layer, input_size : int):
        super(ReLuTransformer, self).__init__()
        self.layer = layer
        self.input_size = input_size

        self.uc = torch.zeros(self.input_size)
        self.uc_b = torch.zeros(self.input_size)
        self.lc = torch.zeros(self.input_size)
        self.lc_b = torch.zeros(self.input_size)
        self.ub = torch.zeros(self.input_size)
        self.lb = torch.zeros(self.input_size)

        #0 initialisation
        self.beta = torch.nn.Parameter(torch.zeros(self.input_size), requires_grad= True) #learnable parameter to determine the tightiest shape of the constraints
        
    def set_constraints(self, uc : torch.Tensor, uc_b : torch.Tensor, lc : torch.Tensor, lc_b : torch.Tensor, mask : torch.Tensor):
        # For nodes earlier in the network. Not for the previous runs of the network.
        self.uc = torch.where(mask, uc, self.uc.detach())
        self.uc_b = torch.where(mask, uc_b, self.uc_b.detach())
        self.lc = torch.where(mask, lc, self.lc.detach())
        self.lc_b = torch.where(mask, lc_b, self.lc_b.detach())
        
    def forward(self, lower_bounds : torch.Tensor, upper_bounds : torch.Tensor):
        """
        Passes the bounds through the ReLU layer and returns the concrete bounds after the ReLU layer. 
        It should also save the constraints of the layer : 
        - uc: symbolic upper bound constraints (input_size, input_size)
        - uc_b: symbolic upper bound constraints bias (input_size)
        - lc: symbolic lower bound constraints (input_size, input_size)
        - lc_b: symbolic lower bound constraints bias (input_size)

        Arguments:
        - The box bounds over the input domain which will be tested
            - lower_bounds (n_batch,input_size)
            - upper_bounds (n_batch, input_size)
        
        Returns:
        - The concrete bounds after the ReLU layer
            - lower_bounds (n_batch, output_size)
            - upper_bounds (n_batch, output_size)
        """
        #python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt --> not verified
        #python code/verifier.py --net fc_1 --spec test_cases/fc_1/img2_mnist_0.0692.txt --> verified
        #python code/verifier.py --net fc_2 --spec test_cases/fc_2/img0_mnist_0.1086.txt --> not verified
        #python code/verifier.py --net fc_2 --spec test_cases/fc_2/img3_mnist_0.0639.txt --> verified
        #python code/verifier.py --net conv_1 --spec test_cases/conv_1/img0_mnist_0.2302.txt --> not verified
        #python code/verifier.py --net conv_1 --spec test_cases/conv_1/img4_mnist_0.1241.txt --> verified
        
        logger.info(f'ReLU layer propagation... {self.layer}')
        self.lb = lower_bounds.clone()
        self.ub = upper_bounds.clone()
        
        assert self.lb.shape[-1] == self.input_size, f"Lower bounds shape is not correct: {self.lb.shape}, {self.input_size}"
        assert self.ub.shape[-1] == self.input_size, f"Upper bounds shape is not correct: {self.ub.shape}"

        #beta_ = torch.zeros(self.input_size)
        lambda_ = self.ub / (self.ub - self.lb)
        mu_ = self.ub - lambda_ * self.ub
        zeros = torch.zeros(self.input_size)
        ones = torch.ones(self.input_size)

        #SET MASKS FOR THE DIFFERENT CASES
        #when lb >= 0
        mask_lower = self.lb.ge(0)
        
        #when ub <= 0
        mask_upper = self.ub.le(0)
        
        #when lb < 0 and ub > 0
        mask_cross = ~ (mask_lower | mask_upper)

        #Note: torch.clamp quickly sets gradient to 0 after which there is no learning 
        beta_bounded = torch.nn.Sigmoid()(self.beta)

        #SET THE LOWER AND UPPER BOUNDS FOR THE DIFFERENT CASES
        self.lb = torch.where(mask_lower, self.lb, self.lb.detach())
        self.ub = torch.where(mask_lower, self.ub, self.ub.detach())
        self.lb = torch.where(mask_upper, torch.zeros_like(self.lb), self.lb.detach())
        self.ub = torch.where(mask_upper, torch.zeros_like(self.ub), self.ub.detach())
        self.lb = torch.where(mask_cross, beta_bounded * self.lb, self.lb.detach())
        self.ub = torch.where(mask_cross, self.ub, self.ub.detach())

        #SET THE CONSTRAINTS FOR THE DIFFERENT CASES AND SAVE THEM
        #when lb >= 0, we are constrained on the segment xi = xj from the ReLu
        self.set_constraints(uc= ones, 
                            uc_b = zeros, 
                            lc = ones,
                            lc_b = zeros, 
                            mask= mask_lower)
        
        #when ub <= 0, we are constrained on the segment xi = 0 from the ReLu
        self.set_constraints(uc= zeros,
                            uc_b = zeros,
                            lc = zeros,
                            lc_b = zeros,
                            mask= mask_upper)
                
        #when lb < 0 and ub > 0, the tightest shape is upper-constrained by the segment xj = lambda * xi + mu
        #where lambda = ub / (ub - lb) and mu = - lb * lambda
        #and lower-constrained by the segment xj = beta * xi
        #where beta is a learnable parameter
        self.set_constraints(uc  = lambda_,
                            uc_b = mu_,
                            lc = beta_bounded,
                            lc_b = zeros,
                            mask = mask_cross)
        logger.debug(f"relu_forward - cl_conc first 5 values {self.lb[:5]}")
        logger.debug(f"relu_forward - cu_conc first 5 values {self.ub[:5]}")
        return self.lb, self.ub
    
    def backwards(self, uc: torch.Tensor, uc_b: torch.Tensor, lc : torch.Tensor, lc_b : torch.Tensor):
        new_uc, new_uc_b, new_lc, new_lc_b = backsub_relu(uc, uc_b, lc, lc_b, self.uc, self.uc_b, self.lc, self.lc_b)
        return new_uc, new_uc_b, new_lc, new_lc_b 
        
class LeakyReLuTransformer(torch.nn.Module):
    def __init__(self, layer : Union[torch.nn.LeakyReLU, torch.nn.ReLU], input_size : int):
        super(LeakyReLuTransformer, self).__init__()
        self.layer = layer
        self.input_size = input_size

        self.uc = torch.zeros(self.input_size)
        self.uc_b = torch.zeros(self.input_size)
        self.lc = torch.zeros(self.input_size)
        self.lc_b = torch.zeros(self.input_size)
        '''
        This class should act as a ReLU transformer when alpha = 0 
        
        '''
        if isinstance(layer, torch.nn.LeakyReLU):
            self.alpha = self.layer.negative_slope
        else:
            self.alpha = 0.0
        
        self.beta = torch.nn.Parameter(torch.rand(self.input_size), requires_grad= True) #learnable parameter to determine the tightiest shape of the constraints
        self.ub = torch.zeros(self.input_size)
        self.lb = torch.zeros(self.input_size)

    def set_constraints(self, uc : torch.Tensor, uc_b : torch.Tensor, lc : torch.Tensor, lc_b : torch.Tensor, mask : torch.Tensor):
        self.uc = torch.where(mask, uc, self.uc.detach())
        self.uc_b = torch.where(mask, uc_b, self.uc_b.detach())
        self.lc = torch.where(mask, lc, self.lc.detach())
        self.lc_b = torch.where(mask, lc_b, self.lc_b.detach())
    
    def forward(self, lower_bounds : torch.Tensor, upper_bounds : torch.Tensor):
        """
        Passes the bounds through the ReLU layer and returns the concrete bounds after the ReLU layer. 
        It should also save the constraints of the layer : 
        - uc: symbolic upper bound constraints (input_size, input_size)
        - uc_b: symbolic upper bound constraints bias (input_size)
        - lc: symbolic lower bound constraints (input_size, input_size)
        - lc_b: symbolic lower bound constraints bias (input_size)

        Arguments:
        - The box bounds over the input domain which will be tested
            - lower_bounds (n_batch, n_channels, height, width)
            - upper_bounds (n_batch, n_channels, height, width)
        
        Returns:
        - The concrete bounds after the ReLU layer
            - lower_bounds (n_batch, output_size)
            - upper_bounds (n_batch, output_size)
        """
        #NETWORKS WITH RELU
        #python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt --> not verified
        #python code/verifier.py --net fc_1 --spec test_cases/fc_1/img2_mnist_0.0692.txt --> verified      

        # NETWORKS WITH LEAKY RELU
        #alpha < 1
        #python code/verifier.py --net fc_3 --spec test_cases/fc_3/img0_mnist_0.1394.txt --> not verified
        #python code/verifier.py --net fc_3 --spec test_cases/fc_3/img4_mnist_0.0780.txt --> verified
        #alpha > 1
        #python code/verifier.py --net fc_4 --spec test_cases/fc_4/img0_mnist_0.2096.txt --> not verified
        #python code/verifier.py --net fc_4 --spec test_cases/fc_4/img4_mnist_0.0554.txt --> verified
        
        #This shouldn't be detached because it comes from before in the network, not from a previous epoch
        prev_ub = upper_bounds.clone()
        prev_lb = lower_bounds.clone()
        logger.info(f'Leaky ReLU layer propagation... {self.layer}')
        logger.debug(f"Leaky ReLU input lower bounds first 50 values {prev_lb[:50]}, input upper bounds first 50 values {prev_ub[:50]}")
        logger.debug(f"Difference between upper and lower bounds {(prev_ub - prev_lb)[0:50]}")
        
        zeros = torch.zeros(self.input_size)
        ones = torch.ones(self.input_size)

        #SET MASKS FOR THE DIFFERENT CASES
        #when lb >= 0
        mask_lower = prev_lb.ge(0)

        #When ub <= 0 
        mask_upper = prev_ub.le(0)

        #When lb < 0 and ub > 0
        mask_cross = ~ (mask_lower | mask_upper)

        #SET THE LOWER AND UPPER BOUNDS FOR THE DIFFERENT CASES
        #Exact case lb >= 0 -> xi <= xj <= xi
        self.lb = torch.where(mask_lower, prev_lb, self.lb.detach())
        self.ub = torch.where(mask_lower, prev_ub, self.ub.detach())

        #Exact case ub <= 0 -> alpha * xi <= xj <= alpha * xi
        self.lb = torch.where(mask_upper, self.alpha * prev_lb, self.lb.detach())
        self.ub = torch.where(mask_upper, self.alpha * prev_ub, self.ub.detach())
        
        self.set_constraints(uc= ones, 
                            uc_b = zeros, 
                            lc = ones,
                            lc_b = zeros, 
                            mask= mask_lower)
        

        self.set_constraints(uc= self.alpha*ones,
                            uc_b = zeros,
                            lc = self.alpha*ones,
                            lc_b = zeros,
                            mask= mask_upper)
        
        #Beta should be between alpha and 1
        beta_bounded = None
        
        #Crossing and alpha < 1 -> Convex shape lower bounded by beta line
        if self.alpha < 1:
            beta_bounded = torch.functional.F.sigmoid(self.beta) * (1 - self.alpha) + self.alpha
            # beta * xi <= xj <= lambda * xi + mu
            self.lb = torch.where(mask_cross, beta_bounded * prev_lb, self.lb.detach())
            self.ub = torch.where(mask_cross, prev_ub, self.ub.detach())

            #Here this should be fixed to used the prev_ub and prev_lb
            lambda_ = (prev_ub - self.alpha * prev_lb) / (prev_ub - prev_lb)
            mu_ = prev_ub - lambda_ * prev_ub
           
            self.set_constraints(uc= lambda_,
                    uc_b = mu_,
                    lc = beta_bounded,             
                    lc_b = zeros,
                    mask= mask_cross)

        #SET THE CONSTRAINTS FOR THE DIFFERENT CASES AND SAVE THEM        
        #Exact case, this is linear transformer
        elif self.alpha == 1:
            self.ub = torch.where(mask_cross, prev_ub, self.ub.detach())
            self.lb = torch.where(mask_cross, prev_lb, self.lb.detach())
            self.set_constraints(
                uc= ones,
                uc_b = zeros,
                lc = ones,
                lc_b = zeros,
                mask= mask_cross
            )
            
        #Crossing and alpha > 1 -> concave shape upper bounded by beta line
        elif self.alpha > 1:
            beta_bounded = torch.functional.F.sigmoid(self.beta) * (self.alpha - 1) + 1

            # lambda * xi + mu <= xj <= beta * xi
            self.lb = torch.where(mask_cross, self.alpha * prev_lb, self.lb.detach())
            self.ub = torch.where(mask_cross, beta_bounded * prev_ub, self.ub.detach())
            
            #Compute the lower constraint
            lambda_ = (prev_ub - self.alpha * prev_lb) / (prev_ub - prev_lb)

            # lambda * ui + mu = ui
            mu_ = (1 - lambda_) * prev_ub

            self.set_constraints(uc= beta_bounded,
                                uc_b = zeros,
                                lc = lambda_,
                                lc_b = mu_,
                                mask= mask_cross)
            
        #We then find which cases these belong to between mask_lower, mask_upper and mask_cross
        mask_lower = mask_bug & mask_lower
        mask_upper = mask_bug & mask_upper
        mask_cross = mask_bug & mask_cross
        
        return self.lb, self.ub
    
    def backwards(self, uc: torch.Tensor, uc_b: torch.Tensor,  lc: torch.Tensor, lc_b : torch.Tensor):
        return backsub_relu(uc, uc_b, lc, lc_b, self.uc, self.uc_b, self.lc, self.lc_b) 
