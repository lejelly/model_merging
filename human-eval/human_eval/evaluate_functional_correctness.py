import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

import re
import os

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    outputpath = os.path.join(os.path.dirname(sample_file) ,"result")
    os.makedirs(outputpath, exist_ok=True)
    out_file = os.path.join(outputpath,"results.txt")
    pattern = r"(seed\d+)"
    match = re.search(pattern, sample_file)
    #params = match.group(1)
    with open(out_file, "a" if os.path.exists(out_file) else "w") as file:
        if os.path.exists(out_file):
            file.write("\n")
        file.write(f"prams: , pass@1: {results}")
    #print(results)

def main():
    fire.Fire(entry_point)


sys.exit(main())
