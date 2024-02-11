import unittest
import torch
from deeppoly import DeepPoly,ReLuTransformer
import argparse
import sys
import logging

logger = logging.getLogger(__name__)


class TestIntegration(unittest.TestCase):


    #Setting up model from the deeppoly paper
    def setUp(self):
        self.paper_model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 1)
        )


        #Init layers with paper weights and biases

        #Linear layer 1
        self.paper_model[0].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        self.paper_model[0].bias.data = torch.tensor([0.0, 0.0])

        #Nothin  to init for RELU

        #Linear layer 2
        self.paper_model[2].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        self.paper_model[2].bias.data = torch.tensor([0.0, 0.0])

        #Nothing to init for RELU

        #Linear layer 3
        self.paper_model[4].weight.data = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        self.paper_model[4].bias.data = torch.tensor([1.0, 0.0])

        
        #Our network includes the verification layer
        self.paper_model[5].weight.data = torch.tensor([[1.0, -1.0]])
        self.paper_model[5].bias.data = torch.tensor([0.0])
        

        self.dp = DeepPoly(self.paper_model, input_size = 2)

        self.lower_bounds_init = torch.tensor([-1.0, -1.0])
        self.upper_bounds_init = torch.tensor([1.0, 1.0])

    
    def test_forward_paper_model(self):
        logger.info("---------Running forward test------------")
        


        out_lb, out_ub = self.dp(self.lower_bounds_init, self.upper_bounds_init)

        #Check if the bounds are correct
        expected_lb = torch.tensor([-1.0])
        expected_ub = torch.tensor([7.0])

        forward_expected_ub_l1 = [2.0, 2.0]
        forward_expected_lb_l1 = [-2.0, -2.0]

        forward_expected_ub_l2 = [2.0, 2.0]
        forward_expected_lb_l2 = [0.0, 0.0]

        forward_expected_ub_l3 = [4.0, 2.0]
        forward_expected_lb_l3 = [0.0, -2.0]

        forward_expected_ub_l4 = [4.0, 2.0]
        forward_expected_lb_l4 = [0.0, 0.0]

        forward_expected_ub_l5 = [7.0, 2.0]
        forward_expected_lb_l5 = [1.0, 0.0]

        #forward_expected_ub_l6 = [7.0]
        forward_expected_lb_l6 = [-1.0]

        print(f"Ver net: {self.dp.verifier_net}")
        print(f"Out ub: {out_ub}")
        print(f"Out lb: {out_lb}")
        print(f"Expected lb: {torch.tensor(forward_expected_lb_l1)}")
        print(f"Actual lb idx 0: {self.dp.verifier_net[0].lb}")

        print(f"Actual lb idx 1: {self.dp.verifier_net[1].lb}")
        print(f"Actual lb idx 2: {self.dp.verifier_net[2].lb}")
        print(f"Actual lb idx 3: {self.dp.verifier_net[3].lb}")
        print(f"Actual lb idx 4: {self.dp.verifier_net[4].lb}")

        print(f"Actual ub idx 0: {self.dp.verifier_net[0].ub}")
        print(f"Actual ub idx 1: {self.dp.verifier_net[1].ub}")
        print(f"Actual ub idx 2: {self.dp.verifier_net[2].ub}")
        print(f"Actual ub idx 3: {self.dp.verifier_net[3].ub}")
        print(f"Actual ub idx 4: {self.dp.verifier_net[4].ub}")

        print(f"Out ub: {out_ub}") 
        print(f"Out lb: {out_lb}")




        print(f"Layer idx 0 - {self.dp.verifier_net[0]}")
        self.assertTrue(torch.allclose(self.dp.verifier_net[0].lb, torch.tensor(forward_expected_lb_l1)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[0].ub, torch.tensor(forward_expected_ub_l1)))

        self.assertTrue(torch.allclose(self.dp.verifier_net[1].lb, torch.tensor(forward_expected_lb_l2)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[1].ub, torch.tensor(forward_expected_ub_l2)))

        self.assertTrue(torch.allclose(self.dp.verifier_net[2].lb, torch.tensor(forward_expected_lb_l3)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[2].ub, torch.tensor(forward_expected_ub_l3)))

        self.assertTrue(torch.allclose(self.dp.verifier_net[3].lb, torch.tensor(forward_expected_lb_l4)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[3].ub, torch.tensor(forward_expected_ub_l4)))

        self.assertTrue(torch.allclose(self.dp.verifier_net[4].lb, torch.tensor(forward_expected_lb_l5)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[4].ub, torch.tensor(forward_expected_ub_l5)))

        self.assertTrue(torch.allclose(self.dp.verifier_net[5].lb, torch.tensor(forward_expected_lb_l6)))
        self.assertTrue(torch.allclose(self.dp.verifier_net[5].ub, torch.tensor(forward_expected_ub_l6)))

        
    

    def test_verification_x13(self):

        logger.info("-------Running backsubstitution test----------")

        #Forward pass
        curr_lower_bounds, curr_upper_bounds = self.dp(self.lower_bounds_init, self.upper_bounds_init)

        #Backsubstitute
        for i in range(len(self.dp.verifier_net) - 2, -1, -1):

            print(f"Backsubstitute layer {i}")

            print(f"Lower bounds: {self.dp.verifier_net[i].lb}, upper bounds : {self.dp.verifier_net[i].lb}")
            curr_lower_bounds, curr_upper_bounds = self.dp.backsubstitute(i)
            

            assert (curr_lower_bounds <= curr_upper_bounds).all(), f"Lower bounds are not smaller than upper bounds after backsubstitution on i: {i}"

            #print(f"Lower bounds: {curr_lower_bounds}, upper bounds : {curr_upper_bounds}")
        



        
        #Check if the bounds are correct
        expected_lb = torch.tensor([1.0])
        expected_ub = torch.tensor([4.0])


        self.assertTrue(torch.allclose(curr_lower_bounds, expected_lb))
        self.assertTrue(torch.allclose(curr_upper_bounds, expected_ub))

    def test_verification_x11_x12(self):

        logger.info("-------Running backsubstitution test----------")

        self.dp.verifier_net = self.dp.verifier_net[:-2]
        #Forward pass
        curr_lower_bounds, curr_upper_bounds = self.dp(self.lower_bounds_init, self.upper_bounds_init)

        #Backsubstitute
        for i in range(len(self.dp.verifier_net) - 2, -1, -1):

            print(f"Backsubstitute layer {i}")

            print(f"Lower bounds: {self.dp.verifier_net[i].lb}, upper bounds : {self.dp.verifier_net[i].lb}")
            curr_lower_bounds, curr_upper_bounds = self.dp.backsubstitute(i)
            

            assert (curr_lower_bounds <= curr_upper_bounds).all(), f"Lower bounds are not smaller than upper bounds after backsubstitution on i: {i}"

            #print(f"Lower bounds: {curr_lower_bounds}, upper bounds : {curr_upper_bounds}")
        


        
        
        #Check if the bounds are correct
        expected_lb = torch.tensor([1.0, 0.0])
        expected_ub = torch.tensor([5.5, 2.0])


        self.assertTrue(torch.allclose(curr_lower_bounds, expected_lb))
        self.assertTrue(torch.allclose(curr_upper_bounds, expected_ub))

    """"
    def test_intermediary_backsubstitution(self):

        for i in range(len(self.paper_model)):

            net = self.paper_model[:i]

            logger.info(f"Testing intermediary backsubstitution on layer {i}")

            dp = DeepPoly(net, input_size = 2)

            curr_lower_bounds, curr_upper_bounds = dp(self.lower_bounds_init, self.upper_bounds_init)



            for i in range(len(dp.verifier_net) - 2, -1, -1):
                logger.info(f"Backsubstitute layer {i}")
    """


  








def main(): 
    print("Application started")
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument('-t', '--test', dest='test', help="Run tests", action='store_true')

    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='WARNING')

    args, unittest_args = parser.parse_known_args()
    

    logger.info("Application started")
    logging.basicConfig(filename= 'test.log', filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')


    unittest.main(argv=sys.argv[:1] + unittest_args)

if __name__ == '__main__':
    
    main()
    