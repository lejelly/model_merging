import re
import sys
import jsonlines
from fraction import Fraction


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0' or abs(float(denominator)) == float('inf'):
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def extract_answer_number_for_ja_mgsm(completion):
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))

def compute_score_for_ja_mgsm(results, lang_detect):
    corrects = []
    incorrect = []
    for question, response, pred, answer in zip(results["question"], results["response"], results["prediction"], results["answer"]):
        if answer == pred:
            corrects.append(True)
        else:
            corrects.append(False)
            incorrect.append({
                "question": question,
                "response": response,
                "prediction": pred,
                "answer": answer
            })
            
    res_dict = {"acc": sum(corrects) / len(corrects)}
    # detect Japanese by fasttext and replace empty string if it's not Ja
    if lang_detect:
        corrects_ja = []
        for i, resp in enumerate(results["response"]):
            res = lang_detect(resp)
            corrects_ja.append(corrects[i] and res.get("__label__ja", 0.0) > 0.5)
        res_dict["acc_ja"] = sum(corrects_ja) / len(corrects_ja)
    return res_dict, incorrect

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def generate_instruction_following_task_prompt(instruction, is_chat_model=True):
    if is_chat_model:
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"""
    else:
        prompt = f"""{instruction}

### Response:
"""
    return prompt

def get_math_task_prompt(prompt_type="zeroshotcot"):
    zeroshotcot =(
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. ASSISTANT: "
    )

    fewshotcot = (
        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        
        "Examples:\n\n"
        
        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
        "ASSISTANT: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6 trees that were planted. The answer is: 6.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
        "ASSISTANT: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is: 5.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
        "ASSISTANT: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is: 39.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
        "ASSISTANT: Jason had 20 lollipops originally. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is: 8.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
        "ASSISTANT: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is: 9.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
        "ASSISTANT: There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room. The answer is: 29.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
        "ASSISTANT: Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday. The answer is: 33.\n\n"

        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? \n"
        "ASSISTANT: Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left. The answer is: 8.\n\n"
    
        "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions. Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. Let's think step by step. \n"        
        "USER: {instruction} \n"
        "ASSISTANT: "
    )
    
    if prompt_type=="fewshotcot":
        return fewshotcot
    return zeroshotcot


def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION


def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems

def get_ja_math_task_prompt(input_text):
    JA_ALPACA_COT_TEMPLATE = f"""以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を日本語で記述してください。一歩一歩考えましょう。

### 指示:
{input_text}

### 応答:"""
    return JA_ALPACA_COT_TEMPLATE