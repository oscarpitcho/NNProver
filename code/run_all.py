import os

evaluation = False
if evaluation:
    nets = ["fc_1", "fc_2", "fc_3", "fc_4", "fc_5", "fc_6", "fc_7", "conv_1", "conv_2", "conv_3", "conv_4"]
    folder = "preliminary_evaluation_test_cases"
else:  
    nets = ["fc_base", "fc_1", "fc_2", "fc_3", "fc_4", "fc_5", "fc_6", "fc_7", "conv_base", "conv_1", "conv_2", "conv_3", "conv_4"]
    folder = "test_cases"
os.system("conda activate rtai-project")

for net in nets:
    print(f"Evaluating network {net}...")
    specs = os.listdir(f"{folder}/{net}")

    for spec in specs:
        os.system(f"python code/verifier.py --net {net} --spec {folder}/{net}/{spec} -l INFO")
        print("\n")
