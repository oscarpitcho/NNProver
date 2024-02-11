import unittest
import torch
import torch.nn as nn
import numpy as np
import math
from deeppoly import LinearTransformer, ConvTransformer
from utils.loading import concretize_bounds, backsub_relu 
from deeppoly import ReLuTransformer
import argparse

class TestConcretizeBounds(unittest.TestCase):

    def test_basic_functionality(self):
        cu = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        cu_b = torch.tensor([5.0, 6.0])
        cl = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        cl_b = torch.tensor([-5.0, -6.0])
        lb = torch.tensor([1.0, 2.0])
        ub = torch.tensor([3.0, 4.0])

        expected_cu_conc = torch.tensor([4.0, 7])
        expected_cl_conc = torch.tensor([-4.0, -7.0])

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        print(f"Upper concrete bounds: {cu_conc}")
        print(f"Lower concrete bounds: {cl_conc}")  

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds do not match.")

    def test_zero_coefficients(self):
        cu = torch.zeros((2, 2))
        cu_b = torch.tensor([5.0, 6.0])
        cl = torch.zeros((2, 2))
        cl_b = torch.tensor([-5.0, -6.0])
        lb = torch.tensor([1.0, 2.0])
        ub = torch.tensor([3.0, 4.0])

        expected_cu_conc = cu_b.squeeze(-1)
        expected_cl_conc = cl_b.squeeze(-1)

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds with zero coefficients do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds with zero coefficients do not match.")

    def test_dimension_mismatch(self):
        cu = torch.randn((3, 2))
        cu_b = torch.randn((2, 1))  # Mismatch here
        cl = torch.randn((3, 2))
        cl_b = torch.randn((2, 1))
        lb = torch.randn((2, 1))
        ub = torch.randn((2, 1))

        with self.assertRaises(AssertionError):
            cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

 

    def test_paper_linear(self):
        cu = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        cu_b = torch.tensor([0.0, 0.0])
        cl = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        cl_b = torch.tensor([0.0, 0.0])
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])

        expected_cu_conc = torch.tensor([2.0, 2.0])
        expected_cl_conc = torch.tensor([-2.0, -2.0])

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        print(f"Upper concrete bounds: {cu_conc}")
        print(f"Lower concrete bounds: {cl_conc}")  

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds do not match.")

    def test_paper_relu_1(self):
        cl = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        cl_b = torch.tensor([0.0, 0.0])
        cu = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        cu_b = torch.tensor([1.0, 1.0])
        lb = torch.tensor([-2.0, -2.0])
        ub = torch.tensor([2.0, 2.0])

        expected_cu_conc = torch.tensor([2.0, 2.0])
        expected_cl_conc = torch.tensor([0.0, 0.0])

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        print(f"Upper concrete bounds: {cu_conc}")
        print(f"Lower concrete bounds: {cl_conc}")  

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds do not match.")

    def test_paper_linear_2(self):

        cu = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        cu_b = torch.tensor([0.0, 0.0])
        cl = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        cl_b = torch.tensor([0.0, 0.0])
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([2.0, 2.0])

        expected_cu_conc = torch.tensor([4.0, 2.0])
        expected_cl_conc = torch.tensor([0.0, -2.0])

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        print(f"Upper concrete bounds: {cu_conc}")
        print(f"Lower concrete bounds: {cl_conc}")  

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds do not match.")

    def test_paper_linear_3(self):

        cu = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        cu_b = torch.tensor([1.0, 0.0])
        cl = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        cl_b = torch.tensor([1.0, 0.0])
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([4.0, 2.0])

        expected_cu_conc = torch.tensor([7.0, 2.0])
        expected_cl_conc = torch.tensor([1.0, 0.0])

        cu_conc, cl_conc = concretize_bounds(cu, cu_b, cl, cl_b, lb, ub)

        print(f"Upper concrete bounds: {cu_conc}")
        print(f"Lower concrete bounds: {cl_conc}")  

        self.assertTrue(torch.allclose(cu_conc, expected_cu_conc), "Upper bounds do not match.")
        self.assertTrue(torch.allclose(cl_conc, expected_cl_conc), "Lower bounds do not match.")
class TestLinearBacksub(unittest.TestCase):



    def test_basic_functionality(self):

        #We test the following system of constraints:

        """
        2x2 - x3 - 1 <= x4 <= x2 + x3  + 3
        x2 - 3x3 - 2 <= x5 <= x2 - x3 + 1

        with (equalities because of linear layer)

        2x0 -  x1 - 2   <= x2 <= 2x0 -  x1 - 2
        -x0 + 3x1 + 1   <= x3 <= -x0 + 3x1 + 1

        Res after Backsub 

        5x0 -  5x1 - 4  <= x4 <=  x0 + 2x1 + 2
        5x0 - 10x1 - 6 <= x5 <= 3x0 - 4x1 - 2



        
        
        """


        prev_uc = torch.tensor([[1.0, 1.0],
                                [1.0,-1.0]])
        

        prev_uc_bias = torch.tensor([3.0, 1.0])



        prev_lc = torch.tensor([[2.0, -1.0],
                                [1.0, -3.0]])
        
        prev_lc_bias = torch.tensor([-1.0, -1.0])


        linear_transform = torch.tensor([[2.0 , -1.0],
                                         [-1.0, 3.0]])

        linear_transform_bias = torch.tensor([-2.0, 1.0])



        uc, uc_b, lc, lc_b = LinearTransformer.do_backsub(prev_uc, prev_uc_bias, prev_lc, prev_lc_bias, linear_transform, linear_transform_bias)



        expected_uc = torch.tensor([[1.0, 2],
                                    [3.0, -4.0]])
                       
        expected_uc_b = torch.tensor([2.0, -2.0])

        expected_lc = torch.tensor([[5.0, -5.0],
                                    [5.0, -10.0]])
        expected_lc_b = torch.tensor([-6.0, -6.0])


        self.assertTrue(torch.allclose(uc, expected_uc), "Upper constraints coefficients do not match.")
        self.assertTrue(torch.allclose(uc_b, expected_uc_b), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(lc, expected_lc), "Lower constraints coefficients do not match.")
        self.assertTrue(torch.allclose(lc_b, expected_lc_b), "Lower constraints biases do not match.")
    
    def test_valid_input(self):
        """Test the function with valid input tensors."""
        # Define input tensors
        uc = torch.randn(2, 3)
        lc = torch.randn(2, 3)
        uc_b = torch.randn(2)
        lc_b = torch.randn(2)
        layer_constraints = torch.randn(3, 4)
        layer_constraints_b = torch.randn(3)

        # Call the function
        result = LinearTransformer.do_backsub(uc, uc_b, lc, lc_b, layer_constraints, layer_constraints_b)

        # Assert the output shape is as expected
        self.assertEqual(result[0].shape, torch.Size([2, 4]))
        # Add more assertions as needed

    def test_mismatched_tensor_shapes(self):
        """Test the function with mismatched tensor shapes."""
        # Define input tensors with mismatched shapes
        uc = torch.randn(2, 3)
        lc = torch.randn(2, 3)
        uc_b = torch.randn(2)
        lc_b = torch.randn(2)
        layer_constraints = torch.randn(5, 4)  # Mismatched shape
        layer_constraints_b = torch.randn(3)

        # Assert that the function raises an error with mismatched shapes
        with self.assertRaises(RuntimeError):
            LinearTransformer.do_backsub(uc, uc_b, lc, lc_b, layer_constraints, layer_constraints_b)


class TestBacksubRelu(unittest.TestCase):
    def setUp(self) -> None:
        # Setup common test variables
        self.input_size = 2
        self.layer = torch.nn.Linear(self.input_size, self.input_size)
        self.relu_transformer = ReLuTransformer(self.layer, self.input_size)

    def test_basic_functionality(self):


        """
        Starting system:

        2x2 - 3x3 + 1<= x4 <= x2 + 2x3 - 1


        Relu Constraints:

        -x0 + 2 <= x2 <= 3x0  - 1  : ---> x2 = max(0, x0) = relu(x0)
        0x1 + 0 <= x3 <= -4x1 - 1  : ---> x3 = max(0, x1) = relu(x1)

        (Lower constraint for x3 is 0)

        Expected result after backsub:

        -2x0 + 12x1 + 8 <= x4 <= 3x0 - 8x1 - 4
        
        """


        uc = torch.tensor([[1.0, 2.0]])
        uc_b = torch.tensor([-1.0])

        lc = torch.tensor([[2.0, -3.0]])
        lc_b = torch.tensor([1.0])

        lambda_u = torch.tensor([3.0, -4.0])
        mu_u = torch.tensor([-1.0, -1.0])

        lambda_l = torch.tensor([-1.0, 0.0])
        mu_l = torch.tensor([2.0, 0.0])


        expected_uc = torch.tensor([[3.0, -8.0]])
        expected_uc_b = torch.tensor([-4.0])

        expected_lc = torch.tensor([[-2.0, 12.0]])
        expected_lc_b = torch.tensor([8.0])

        cu, cu_b, cl, cl_b = backsub_relu(uc, uc_b, lc, lc_b, lambda_u, mu_u, lambda_l, mu_l)



        print(f"Upper constraints bias: {cu_b}")
        print(f"Expected upper constraints bias: {expected_uc_b}")

        print(f"Upper constraints coefficients: {cu}")
        print(f"Expected upper constraints coefficients: {expected_uc}")

        print(f"Lower constraints bias: {cl_b}")
        print(f"Expected lower constraints bias: {expected_lc_b}")

        print(f"Lower constraints coefficients: {cl}")
        print(f"Expected lower constraints coefficients: {expected_lc}")
        
        self.assertTrue(torch.allclose(cu, expected_uc), "Upper constraints coefficients do not match.")
        self.assertTrue(torch.allclose(cu_b, expected_uc_b), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(cl, expected_lc), "Lower constraints coefficients do not match.")
        self.assertTrue(torch.allclose(cl_b, expected_lc_b), "Lower constraints biases do not match.")


class TestConvTransformer(unittest.TestCase):
    def setUp(self):
        # Setup parameters
        self.in_channels = 3
        self.out_channels = 2
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.input_size = 784 * self.in_channels  # e.g., for an 8x8 input with 3 channels

        conv_layer = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        self.conv_transformer = ConvTransformer(conv_layer, self.input_size)

    def test_convolution(self):
        # Generate a random input tensor
        input_tensor = torch.randn(1, 
                                   self.in_channels,
                                   int(np.sqrt(self.input_size // self.in_channels)), 
                                   int(np.sqrt(self.input_size // self.in_channels)))

        # Convolution using Conv2d
        conv2d_output = self.conv_transformer.layer(input_tensor)

        # Convolution using ConvTransformer
        transformed_input = input_tensor.view(1, -1)
        conv_transformer_output = torch.matmul(self.conv_transformer.constraints, transformed_input.t()).t() + self.conv_transformer.constraints_b
        conv_transformer_output = conv_transformer_output.view(1, self.conv_transformer.layer.out_channels, self.conv_transformer.out_height, self.conv_transformer.out_width)

        # Compare the outputs
        self.assertTrue(torch.allclose(conv2d_output, conv_transformer_output, atol=1e-6))
    


class TestReluTransformer(unittest.TestCase):
    """
    
    
    """
    def setUp(self) -> None:
        # Setup common test variables
        self.input_size = 2
        self.layer = torch.nn.Linear(self.input_size, self.input_size)
        self.relu_transformer = ReLuTransformer(self.layer, self.input_size)
        self.zeros = torch.zeros(self.input_size)
        self.ones = torch.ones(self.input_size)

    def test_forward_lb_ge_zero(self):
        # Test case where lower bounds are >= 0


        # BOUNDS
        lower_bounds = torch.Tensor([1.0, 2.0])
        upper_bounds = torch.Tensor([3.0, 4.0])

        lb, ub = self.relu_transformer.forward(lower_bounds, upper_bounds)

        print(f"Lower bounds: {lb}, upper bounds: {ub}")
        expected_lb = torch.Tensor([1.0, 2.0])
        expected_ub = torch.Tensor([3.0, 4.0])
        
        self.assertTrue(torch.allclose(lb, expected_lb), "Lower bounds do not match.")
        self.assertTrue(torch.allclose(ub, expected_ub), "Upper bounds do not match.")


        # CONSTRAINTS

        
        self.assertTrue(torch.allclose(self.relu_transformer.uc, self.ones), "Upper constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc, self.ones), "Lower constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.uc_b, self.zeros), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc_b, self.zeros), "Lower constraints bias do not match.")



    def test_forward_ub_le_zero(self):
        # Test case where upper bounds are <= 0
        upper_bounds = torch.Tensor([-1.0, -2.0])
        lower_bounds = torch.Tensor([-3.0, -4.0])
        lb, ub = self.relu_transformer.forward(lower_bounds, upper_bounds)

        print(f"Lower bounds: {lb}, upper bounds: {ub}")
        expected_lb = torch.Tensor([0.0, 0.0])
        expected_ub = torch.Tensor([0.0, 0.0])
        
        self.assertTrue(torch.allclose(lb, expected_lb), "Lower bounds do not match.")
        self.assertTrue(torch.allclose(ub, expected_ub), "Upper bounds do not match.")

        # CONSTRAINTS

        self.assertTrue(torch.allclose(self.relu_transformer.uc, self.zeros), "Upper constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc, self.zeros), "Lower constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.uc_b, self.zeros), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc_b, self.zeros), "Lower constraints bias do not match.")




    def test_forward_lb_lt_zero_ub_gt_zero(self):
        # Test case where lower bounds < 0 and upper bounds > 0
        lower_bounds = torch.Tensor([-3.0, -2.0])
        upper_bounds = torch.Tensor([1.0, 3.0])
        lb, ub = self.relu_transformer.forward(lower_bounds, upper_bounds)

        print(f"Lower bounds: {lb}, upper bounds: {ub}")
        expected_lb = torch.Tensor([0.0, -2.0])
        expected_ub = torch.Tensor([1.0, 3.0])
        
        self.assertTrue(torch.allclose(lb, expected_lb), "Lower bounds do not match.")
        self.assertTrue(torch.allclose(ub, expected_ub), "Upper bounds do not match.")

        # CONSTRAINTS
        print(f"Upper constraints: {self.relu_transformer.uc}, lower constraints: {self.relu_transformer.lc}")
        print(f"Upper constraints b: {self.relu_transformer.uc_b}, lower constraints b: {self.relu_transformer.lc_b}")

        expected_uc = torch.Tensor([0.25, 0.6]) # lambda_0 = 0.25, lambda_1 = 0.6
        expected_lc = torch.Tensor([0.0, 1.0])
        expected_uc_b = torch.Tensor([0.75, 1.2]) # mu_0 = 0.75, mu_1 = 1.2
        expected_lc_b = torch.Tensor([0.0, 0.0])

        
        self.assertTrue(torch.allclose(self.relu_transformer.uc, expected_uc), "Upper constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc, expected_lc), "Lower constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.uc_b, expected_uc_b), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc_b, expected_lc_b), "Lower constraints bias do not match.")

    def test_mixed_bounds(self):
        if self.input_size != 4:
            self.skipTest("This test is only valid for an input size of 4.")

        lower_bounds = torch.Tensor([-3.0, -2.0, -2.0, 3.0])
        upper_bounds = torch.Tensor([1.0, 3.0, -1.0, 4.0])
        lb, ub = self.relu_transformer.forward(lower_bounds, upper_bounds)

        print(f"Lower bounds: {lb}, upper bounds: {ub}")
        expected_lb = torch.Tensor([0.0, -2.0, 0.0, 3.0])
        expected_ub = torch.Tensor([1.0, 3.0, 0.0, 4.0])

        self.assertTrue(torch.allclose(lb, expected_lb), "Lower bounds do not match.")
        self.assertTrue(torch.allclose(ub, expected_ub), "Upper bounds do not match.")

        # CONSTRAINTS
        print(f"Upper constraints: {self.relu_transformer.uc}, lower constraints: {self.relu_transformer.lc}")
        print(f"Upper constraints b: {self.relu_transformer.uc_b}, lower constraints b: {self.relu_transformer.lc_b}")

        expected_uc = torch.Tensor([0.25, 0.6, 0.0, 1.0])
        expected_lc = torch.Tensor([0.0, 1.0, 0.0, 1.0])
        expected_uc_b = torch.Tensor([0.75, 1.2, 0.0, 0.0])
        expected_lc_b = torch.Tensor([0.0, 0.0, 0.0, 0.0])

        self.assertTrue(torch.allclose(self.relu_transformer.uc, expected_uc), "Upper constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc, expected_lc), "Lower constraints do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.uc_b, expected_uc_b), "Upper constraints bias do not match.")
        self.assertTrue(torch.allclose(self.relu_transformer.lc_b, expected_lc_b), "Lower constraints bias do not match.")


if __name__ == '__main__':
    unittest.main()
