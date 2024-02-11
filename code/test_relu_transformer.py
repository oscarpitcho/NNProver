import unittest
import torch
import torch.nn as nn
import numpy as np
import math
from deeppoly import LinearTransformer, ConvTransformer
from utils.loading import concretize_bounds, backsub_relu 
from deeppoly import ReLuTransformer
import argparse
import sys
import logging

logger = logging.getLogger(__name__)



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
        logger.info("----Testing forward pass with lower bounds >= 0-----")
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
        logger.info("----Testing forward pass with upper bounds <= 0-----")
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
        logger.info("----Testing forward pass with lower bounds < 0 and upper bounds > 0-----")
        # Test case where lower bounds < 0 and upper bounds > 0
        lower_bounds = torch.Tensor([-3.0, -2.0])
        upper_bounds = torch.Tensor([1.0, 3.0])

        logger.info(f"Lower bounds: {lower_bounds}, upper bounds: {upper_bounds}")
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


def main(): 
    print("Application started")
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument('-t', '--test', dest='test', help="Run tests", action='store_true')

    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='WARNING')

    args, unittest_args = parser.parse_known_args()
    

    logger.info("Application started")
    logging.basicConfig(filename= 'test_relu.log', filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')


    unittest.main(argv=sys.argv[:1] + unittest_args)

if __name__ == '__main__':
    
    main()