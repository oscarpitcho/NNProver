# NN Prover

## Project Description
This is a sound formal robustness prover for Neural Networks. it is based on [Sing et al.](https://ggndpsngh.github.io/files/DeepPoly.pdf). Our implementation propagates initial box constraints on the input throughout the network. Additionally to ensure
a high degree of precision, we regularly back substitute constraints through the network and, if required, differentiate our prover to fine-tune it to the case being proved.

## Folder structure
In the directory `code`, you can find 3 files and one folder: 
- The file `networks.py` contains the fully-connected and convolutional neural network architectures used in this project.
- The file `verifier.py` contains a template of the verifier. Loading of the stored networks and test cases is already implemented in the `main` function. If you decide to modify the `main` function, please ensure that the parsing of the test cases works correctly. Your task is to modify the `analyze` function by building upon DeepPoly convex relaxation. Note that the provided verifier template is guaranteed to achieve **0** points (by always outputting `not verified`).
- The file `evaluate` will run all networks and specs currently in the repository. It follows the calling convention we will use for grading.
- The folder `utils` contains helper methods for loading and initialization (There is most likely no need to change anything here).


In the directory `models`, you can find 14 neural networks (9 fully connected and 5 convolutional) weights. These networks are loaded using PyTorch in `verifier.py`. 
In the directory `test_cases`, you can find 13 subfolders (the folder for `fc_6` contains both examples for `cifar10` and `mnist`). Each subfolder is associated with one of the networks using the same name. In a subfolder corresponding to a network, you can find 2 test cases for each network. 

Note that all inputs are images with pixel values between 0 and 1. The same range also applies to all abstract bounds that we want to verify. 

## Setup instructions


```bash
$ virtualenv venv --python=python3.10
$ source venv/bin/activate
$ pip install -r requirements.txt
```

If you prefer conda environments we also provide a conda `environment.yaml` file which you can install (After installing [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html)) via

```bash
$ conda env create -f ./environment.yaml
$ conda activate rtai-project
```

for `mamba` simply replace `conda` with `mamba`.

## Running the verifier

To test your verifier, use the following command template 

```bash
$ python code/verifier.py --net {net} --spec test_cases/{net}/img{id}_{dataset}_{eps}.txt
```

In this command, 
- `net` is equal to one of the following values (each representing one of the networks we want to verify): `fc_base, fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, conv_base, conv_1, conv_2, conv_3, conv_4`.
- `id` is simply a numerical identifier of the case. They are not always ordered as they have been directly sampled from a larger set of cases.
- `dataset` is the dataset name, i.e.,  either `mnist` or `cifar10`.
- `eps` is the perturbation that the verifier should certify in this test case.

For instance
```bash
$ python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt
```

To evaluate the verifier on all networks and sample test cases, we provide an evaluation script.
You can run this script from the root directory using the following commands:

```bash
chmod +x code/evaluate
code/evaluate
```

## References

This project was done for the 2023 edition of the Reliable & Trustworthy AI (RTAI) course @ ETHZ.
