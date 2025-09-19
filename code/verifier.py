import argparse
import torch
from deeppoly import DeepPoly, ReLuTransformer, LeakyReLuTransformer

from networks import get_network
from utils.loading import parse_spec
from time import time

import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

DEVICE = "cpu"
def verify_bounds(lower_bounds, true_label):
    return (lower_bounds[:true_label] > 0).all() and (lower_bounds[true_label + 1:] > 0).all()
    
def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    # input_shape : (n_channels, height, width)

    input_size = inputs.view(-1).size(0)
    
    lower_bounds_init = torch.clamp(
        inputs - eps, min=0.0).to(DEVICE)
    upper_bounds_init = torch.clamp(
        inputs + eps, max=1.0).to(DEVICE)
    
    #We find the number of classes by looking at the output shape of the network.
    nb_classes = net(inputs.unsqueeze(0)).shape[1]

    #Building additional verification layer, new neurons are all defined as o_true - o_i
    #We keep bias for interopperability with other layers
    verification_layer = torch.nn.Linear(nb_classes, nb_classes, bias=True)
    true_label_matrix = torch.zeros(nb_classes, nb_classes)
    true_label_matrix[:, true_label] = 1
    layer_weights = true_label_matrix - torch.eye(nb_classes) #o_true - o_i

    verification_layer.weight.data = layer_weights.to(DEVICE)
    verification_layer.bias.data = torch.zeros(nb_classes).to(DEVICE)
    
    net.add_module(f"{len(net)}", verification_layer)
    
    logger.info(f"Network analyzed - {net}")
    dp = DeepPoly(net, input_size)

    train_beta = any([isinstance(layer, ReLuTransformer) or isinstance(layer, LeakyReLuTransformer) for layer in dp.verifier_net])

    #################################
    '''TRAINING OF BETA PARAMETER '''
    lr = 0.8
    epochs = 20
    if train_beta:
        logger.info("---------Training beta parameter------------")
        torch.autograd.set_detect_anomaly(True)
        
        #We freeze the weights of the network and only optimize the beta parameter of the relu transformer.
        for param in dp.verifier_net.parameters():
            param.requires_grad = False
        
        for layer in dp.verifier_net:
            if isinstance(layer, ReLuTransformer) or isinstance(layer, LeakyReLuTransformer):
                layer.beta.requires_grad = True
                
        optimizer = torch.optim.Adam(
                    dp.verifier_net.parameters(), 
                    lr=lr # Learning rate
                )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        for epoch in range(epochs):
            logger.info(f"#### Epoch {epoch}")

            ver_lb, _ = dp(lower_bounds_init, upper_bounds_init)
            is_verified = verify_bounds(ver_lb, true_label)
            
            if is_verified:
                print(f"Stopped after {epoch} epochs - after backsub")
                return is_verified

            # Select all activations except true label
            invalid_classes_activations = torch.cat((ver_lb[:true_label], ver_lb[true_label + 1:]), dim=0)

            logger.info(f"Invalid classes activations: {invalid_classes_activations}")
            loss = - torch.sum(invalid_classes_activations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            logger.info(f"Epoch {epoch} - loss: {loss.item()}")
         
        logger.info("---------Training beta parameter finished------------")
    ver_lb, _ = dp(lower_bounds_init, upper_bounds_init)
    is_verified = verify_bounds(ver_lb, true_label)

    return is_verified
    
def main():
    start = time()

    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='WARNING')
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)
    
    logger.info("Application started")
    logging.basicConfig(filename= 'output.log', filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')

    print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
        logger.info("verified")
    else:
        print("not verified")
        logger.info("not verified")
    print(f"Time: {time() - start} seconds")

    
if __name__ == "__main__":



    main()
