import re
import os
import matplotlib.pyplot as plt
import argparse

def draw(MODEL_NAME, METHOD_NAME):
    # Path to the uploaded text file
    file_path = f'/work/gb20/b20042/model_merging/results/single_model_inference/ja_mgsm/{METHOD_NAME}/{MODEL_NAME}.txt'
    fig_name = f'/work/gb20/b20042/model_merging/figs/{METHOD_NAME}/{MODEL_NAME}.png'
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    # Lists to store the extracted accuracy and drop rate values
    accuracies = []
    ja_accuracies = []
    drop_rates = []

    # Read the file and extract the data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Extract accuracy and drop_rate using regular expressions
            accuracy_match = re.search(r'accuracy: ([\d\.]+)', line)
            ja_accuracy_match = re.search(r'accuracy_ja: ([\d\.]+)', line)
            drop_rate_match = re.search(r'drop_rate: ([\d\.]+)', line)

            # Append the extracted values to the lists, handling 'xxxx' as None for accuracy
            if accuracy_match and drop_rate_match:
                accuracies.append(float(accuracy_match.group(1)))
                ja_accuracies.append(float(ja_accuracy_match.group(1)))
                drop_rates.append(float(drop_rate_match.group(1)))
    
    sorted_data = sorted(zip(drop_rates, accuracies, ja_accuracies))
    drop_rates, accuracies, ja_accuracies = zip(*sorted_data)
    drop_rates = list(drop_rates)
    accuracies = list(accuracies)
    ja_accuracies = list(ja_accuracies)

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
    plt.ylim(bottom=0, top=1.0)
    plt.plot(drop_rates, accuracies, 'b-', label='Response in Any Language', linewidth=3)
    plt.plot(drop_rates, accuracies, 'bo', markersize=5)
    plt.plot(drop_rates, ja_accuracies, 'r-', label='Response in Only Japanese')
    plt.plot(drop_rates, ja_accuracies, 'ro', markersize=5)
    plt.xlabel('Drop Out Rate of Delta Parameter\n(Delta Parameter θ_delta = θ_finetuned - θ_pretrained)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Performance on MGSM-JA w/{METHOD_NAME}')
    plt.grid(True)
    plt.legend()

    # Save the plot to a file
    plt.savefig(fig_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="alpaca_eval", help="dataset to be used", choices=["WizardLMTeam/WizardMath-7B-V1.1", "augmxnt/shisa-gamma-7b-v1", "GAIR/Abel-7B-002", "tokyotech-llm/Swallow-MS-7b-v0.1"])
    parser.add_argument("--method_name", type=str, default="alpaca_eval", help="dataset to be used", choices=["dare", "droponly", "finetuned", "magnitude"])
    args = parser.parse_args()
    
    draw(args.model_name, args.method_name)
    