import unittest
from deeppoly import ReLuTransformer, DeepPoly
import torch
import argparse
import sys
import logging

logger = logging.getLogger(__name__)

class PaperModelTests(unittest.TestCase):
    

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



    def test_requires_grad_lb_ub_with_backsub(self):

        out_lb, out_ub = self.dp(self.lower_bounds_init, self.upper_bounds_init)


        self.assertTrue(out_lb.requires_grad)
        self.assertTrue(out_ub.requires_grad)

        #We now proceed to backsubstitute one layer after another. The output result should always require grad
        for i in range(len(self.dp.verifier_net) - 2, -1, -1):
            out_lb, out_ub = self.dp.backsubstitute(i)
            self.assertTrue(out_lb.requires_grad)
            self.assertTrue(out_ub.requires_grad)


    def test_differentiability_no_backsub(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75

        for param in self.dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in self.dp.verifier_net:
            if isinstance(layer, ReLuTransformer):
                layer.beta.requires_grad = True
            

        optimizer = torch.optim.Adam(
                    self.dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
         

        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            ver_lb, ver_ub = self.dp(self.lower_bounds_init, self.upper_bounds_init)

            loss = -torch.sum(ver_lb)
            print(f"Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_differentiability_with_backsub(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75

        for param in self.dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in self.dp.verifier_net:
            if isinstance(layer, ReLuTransformer):
                layer.beta.requires_grad = True
            

        optimizer = torch.optim.Adam(
                    self.dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
         

        ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)

            for i in range(len(self.dp.verifier_net) - 2, -1, -1):
                print(f"Backsubstitute layer {i}")
                ver_lb, _ = self.dp.backsubstitute(i)


            loss = -torch.sum(ver_lb)
            print(f"Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test_differentiability_with_backsub_intermediary(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75



        for j in range(len(self.dp.verifier_net) - 2, -1, -1):
            print(f"Running with backsub down to layer {j}")

            #We reset the model to the initial state
            self.setUp()

            for param in self.dp.verifier_net.parameters():
                param.requires_grad = False
        
            for layer in self.dp.verifier_net:
                if isinstance(layer, ReLuTransformer):
                    layer.beta.requires_grad = True
                

            optimizer = torch.optim.Adam(
                        self.dp.verifier_net.parameters(), 
                        lr=lr # Learning rate
                    )
            
            for epoch in range(epochs):
                print(f"Epoch {epoch}")
                ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)
                for i in range(len(self.dp.verifier_net) - 2, j+1, -1):
                    print(f"Backsubstitute being run on layer {i}")
                    ver_lb, _ = self.dp.backsubstitute(i)


                loss = -torch.sum(ver_lb)
                print(f"Loss: {loss}")

                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()



class BasicModelTests(unittest.TestCase):

#Setting up model from the deeppoly paper
    def setUp(self):
        self.paper_model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2)
        )


        #Init layers with paper weights and biases

        #Linear layer 1
        self.paper_model[0].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        self.paper_model[0].bias.data = torch.tensor([0.0, 0.0])

        #Nothin  to init for RELU

        #Linear layer 2
        self.paper_model[2].weight.data = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        self.paper_model[2].bias.data = torch.tensor([0.0, 0.0])

        self.dp = DeepPoly(self.paper_model, input_size = 2)

        self.lower_bounds_init = torch.tensor([-1.0, -1.0])
        self.upper_bounds_init = torch.tensor([1.0, 1.0])


    def test_requires_grad_lb_ub_with_backsub(self):

        out_lb, out_ub = self.dp(self.lower_bounds_init, self.upper_bounds_init)


        self.assertTrue(out_lb.requires_grad)
        self.assertTrue(out_ub.requires_grad)

        #We now proceed to backsubstitute one layer after another. The output result should always require grad
        for i in range(len(self.dp.verifier_net) - 2, -1, -1):
            out_lb, out_ub = self.dp.backsubstitute(i)
            self.assertTrue(out_lb.requires_grad)
            self.assertTrue(out_ub.requires_grad)


    def test_differentiability_no_backsub(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75

        for param in self.dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in self.dp.verifier_net:
            if isinstance(layer, ReLuTransformer):
                layer.beta.requires_grad = True
            

        optimizer = torch.optim.Adam(
                    self.dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
         

        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            ver_lb, ver_ub = self.dp(self.lower_bounds_init, self.upper_bounds_init)

            loss = -torch.sum(ver_lb)
            print(f"Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_differentiability_with_backsub(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75

        for param in self.dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in self.dp.verifier_net:
            if isinstance(layer, ReLuTransformer):
                layer.beta.requires_grad = True
            

        optimizer = torch.optim.Adam(
                    self.dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
         

        ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)
            for i in range(len(self.dp.verifier_net) - 2, -1, -1):
                print(f"Backsubstitute layer {i}")
                ver_lb, _ = self.dp.backsubstitute(i)


            loss = -torch.sum(ver_lb)
            print(f"Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test_differentiability_with_backsub_intermediary(self):
        lr = 0.7
        epochs = 10
        step_size = 1
        gamma = 0.75



        for j in range(len(self.dp.verifier_net) - 2, -1, -1):
            print(f"Running with backsub down to layer {j}")

            #We reset the model to the initial state
            self.setUp()

            for param in self.dp.verifier_net.parameters():
                param.requires_grad = False
        
            for layer in self.dp.verifier_net:
                if isinstance(layer, ReLuTransformer):
                    layer.beta.requires_grad = True
                

            optimizer = torch.optim.Adam(
                        self.dp.verifier_net.parameters(), 
                        lr=lr # Learning rate
                    )
            


            for epoch in range(epochs):
                print(f"Epoch {epoch}")
                ver_lb, _ = self.dp(self.lower_bounds_init, self.upper_bounds_init)

                for i in range(len(self.dp.verifier_net) - 2, j+1, -1):
                    print(f"Backsubstitute being run on layer {i}")
                    ver_lb, _ = self.dp.backsubstitute(i)


                loss = -torch.sum(ver_lb)
                print(f"Loss: {loss}")

                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()



class TestDifferentiationExact_1(unittest.TestCase):
    
    @staticmethod
    def verify_bounds(lower_bounds, true_label):
        return (lower_bounds[:true_label] > 0).all() and (lower_bounds[true_label + 1:] > 0).all()

    def test_differentiability_with_backsub_intermediary(self):

        model = torch.nn.Sequential(
            torch.nn.Relu(),
            torch.nn.Linear(4, 2))
        
        #crossing bounds, with what should be two diffent
        lower_bounds_init = torch.Tensor([-3.0, -2.0, -2.0, 3.0])
        upper_bounds_init = torch.Tensor([1.0, 3.0, -1.0, 4.0])

        """
        Network definition 

        x1 in [-3.0 , 1.0]
        x2 = [-2.0, 3.0]
        x3 = [-2.0, -1.0]
        x4 = [3.0, 4.0]


        x5 = Relu(x1)
        x6 = Relu(x2)
        x7 = Relu(x3)
        x8 = Relu(x4)


        x9  = x5 + x6 - x7 - x8
        x10 = x5 - x6 + x7 - x8

        

        #x9 is true label
        x11 = x9.lb - x10.ub --> We want x11 > 0
        
        
        Constraints definitions 


        0.5 * x1 <=  x5 <= 0.25 * x1 + 0.75
        0.5 * x2 <=  x6 <= 0.6 * x2 + 1.2
        0        <=  x7 <= 0
        x4       <=  x8 <= x4


        Backsubstitution equations

        beta1 * x1 +  beta2 * x2 + 0 - x4  <= x9  <= 0.25 * x1 + 0.75 + 0.6 * x2 + 1.2 + 0 - x4

        beta1 * x1 - 0.6*x2 + 1.2 + 0 - x4 <= x10 <=  0.25 * x1 + 0.75 - beta2 * x2 + 0 - x4

        x11 = 0.25 * x1 + 0.75 + 0.6 * x2  - (beta1 * x1 - 0.6*x2)
            = (0.25 + beta1) * x1 + (0.75 - 0.6) * x2 + 0.75
        

            


        

        
        """
        #Linear layer 1

        model[1].weight.data = torch.tensor([[1.0, 1.0, -1.0, -1.0],
                                            [1.0, -1.0, 1.0, -1.0]])
        

        model[1].bias.data = torch.tensor([0.0, 0.0])
        

        dp = DeepPoly(model, input_size= 2)
        

        expected_uc = torch.Tensor([0.25, 0.6, 0.0, 1.0])
        expected_lc = torch.Tensor([0.5, 0.5, 0.0, 1.0])
        expected_uc_b = torch.Tensor([0.75, 1.2, 0.0, 0.0])
        expected_lc_b = torch.Tensor([0.0, 0.0, 0.0, 0.0])

        #################################
        '''TRAINING OF BETA PARAMETER '''
        lr = 0.7
        epochs = 20
        step_size = 1
        gamma = 0.8



        logger.info("---------Training beta parameter------------")

        torch.autograd.set_detect_anomaly(True)


        #We freeze the weights of the network and only optimize the beta parameter of the relu transformer.
        for param in dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in dp.verifier_net:
            if isinstance(layer, ReLuTransformer):
                layer.beta.requires_grad = True
            

        optimizer = torch.optim.Adam(
                    dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            logger.info(f"#### Epoch {epoch}")

            ver_lb, ver_ub = dp(lower_bounds_init, upper_bounds_init)
            
            #Look if the network is verified after forward pass
            is_verified = verify_bounds(ver_lb, true_label)
            if is_verified:
                print(f"Verified after {epoch} epochs - after forward")
                break
            
            #Look if the network is verified each step of  backsubstitution
                
            
            
            for i in range(len(dp.verifier_net) - 2, -1, -1):
                ver_lb, ver_ub = dp.backsubstitute(i)
                is_verified = verify_bounds(ver_lb, true_label)

                logger.info(f"Backsubstitution done of layer {i} done, examining gradients intermediate layers")
                logger.debug(f"ver_ub requires grad: {ver_ub.requires_grad}")
                logger.debug(f"ver_lb requires grad: {ver_lb.requires_grad}")

            if is_verified:
                print(f"Verified after {epoch} epochs - after backsub")
                break
            
            

            invalid_classes_activations = torch.cat((ver_lb[:true_label], ver_lb[true_label + 1:]), dim=0)

            logger.info(f"Invalid classes activations: {invalid_classes_activations}")

            #Losses tried

            #Minimize the total missclassification

            loss = -torch.sum(invalid_classes_activations)

            #Minimize the _smallest_ missclassification. Reason is this would be easier to optimize one at a time.
            #loss = torch.functional.F.relu(-invalid_classes_activations).min()


            print(f"Loss: {loss.item()}")

            optimizer.zero_grad()

            loss.backward()

            with torch.no_grad():
                for layer in dp.verifier_net:
                    if isinstance(layer, ReLuTransformer) :
                        #layer.beta.grad += torch.randn(layer.beta.grad.shape) * 0.01

                        logger.info(f"Backwards done, examining gradients intermediate layers {layer}")

                        logger.info(f"Beta Gradient norm:  {layer.beta.grad.norm()}") 
                        logger.info(f"Beta Gradient:  {layer.beta.grad}")  # gives None meaning that something is broken in the computation graph
                        logger.info(f"Beta vector norm:  {layer.beta.norm()}")
                        logger.info(f"Beta vector:  {layer.beta}")  # gives None meaning that something is broken in the computation graph
                        logger.info(f"Beta bounded: {torch.functional.F.sigmoid(layer.beta)}")
            
            #print(dp.verifier_net[3].beta.grad)  # gives None meaning that something is broken in the computation graph
            
            optimizer.step()
            scheduler.step(loss)


            logger.info(f"Epoch {epoch} - loss: {loss.item()}")
            logger.info(f"beta: {dp.verifier_net[1].beta[:5]}")
            logger.info(f"Lower bounds : {ver_lb[:5]}")




        logger.info("---------Training beta parameter finished------------")
        #################################





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