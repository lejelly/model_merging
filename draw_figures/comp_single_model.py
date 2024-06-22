import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

def draw(MODEL_NAME, METHOD_NAME):
    # Path to the uploaded text file
    file_path = f'/p-shared-11/model_merging/results/single_model_inference/zeroshotcot/gsm8k_{METHOD_NAME}_{MODEL_NAME}.txt'
    fig_name = f'/p-shared-11/model_merging/figs/gsm8k_{MODEL_NAME}_{METHOD_NAME}.png'
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    # Lists to store the extracted accuracy and drop rate values
    accuracies = []
    drop_rates = []

    # Read the file and extract the data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Extract accuracy and drop_rate using regular expressions
            accuracy_match = re.search(r'accuracy: ([\d\.]+)', line)
            drop_rate_match = re.search(r'drop_rate: ([\d\.]+)', line)

            # Append the extracted values to the lists, handling 'xxxx' as None for accuracy
            if accuracy_match and drop_rate_match:
                accuracies.append(float(accuracy_match.group(1)))
                drop_rates.append(float(drop_rate_match.group(1)))
    
    # Revise METHOD_NAME
    if METHOD_NAME=='dare':
        METHOD_NAME=='<DARE>'
    elif METHOD_NAME=='droponly':
        METHOD_NAME=='<DropOnly>'
    elif METHOD_NAME=='finetuned':
        METHOD_NAME=='<Masking fine-tuned full parameter>'
    elif METHOD_NAME=='magnitude':
        METHOD_NAME=='<magnitude-based pruning>'

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(drop_rates, accuracies, 'o-', label=MODEL_NAME, color='red')
    plt.xlabel('Drop Rate')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Performance on GSM8K w/{METHOD_NAME}')
    plt.grid(True)
    plt.legend()

    # Save the plot to a file
    plt.savefig(fig_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="alpaca_eval", help="dataset to be used", choices=["WizardLMTeam/WizardMath-7B-V1.1", "augmxnt/shisa-gamma-7b-v1", "GAIR/Abel-7B-002", "tokyotech-llm/Swallow-MS-7b-v0.1", "BioMistral/BioMistral-7B"])
    parser.add_argument("--method_name", type=str, default="alpaca_eval", help="dataset to be used", choices=["dare", "droponly", "finetuned", "magnitude"])
    args = parser.parse_args()
    
    draw(args.model_name, args.method_name)
    