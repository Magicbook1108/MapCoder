from typing import List
import tiktoken
import os
import copy
import time

from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results
from utils.parse import parse_response


class BaseStrategy(object):
    def __init__(
        self,
        model: BaseModel,
        data: Dataset,
        language: str,
        pass_at_k: int,
        task_amount: int,
        results: Results,
        name: str,
        verbose: bool = True,
    ):
        self.model = model
        self.data = data
        self.pass_at_k = pass_at_k
        self.results = results
        self.language = language
        self.verbose = verbose
        self.total_tokens = 0
        self.task_amount = task_amount
        self.path = "./outputs/result"
        self.name = name

    def gpt_chat(self, processed_input: List[dict]) -> (str, int, int):
        return self.model.prompt(processed_input=processed_input)

    def run_single_pass(self, item: dict):
        pass

    def run(self):
        num_items = len(self.data) if self.task_amount == -1 else self.task_amount
        num_success = 0
        self.total_tokens = 0
        for i, item in enumerate(self.data[:num_items]):
            print("", flush=True, end="")

            if i < len(self.results):
                item = copy.deepcopy(self.results[i])
                cur_pass = len(item["source_codes"])
                is_solved = item["is_solved"]
                cur_imp = item["source_codes"][-1]
                self.total_tokens += int(item["completion_tokens"][0]) + int(item["prompt_tokens"][0])
            else:
                item = copy.deepcopy(item)
                item["source_codes"] = []
                item["responses"] = []
                item["prompt_tokens"] = []
                item["completion_tokens"] = []
                item["no_of_try"] = 0

                cur_pass = 0
                is_solved = False
                cur_imp = ""

            while cur_pass < self.pass_at_k and not is_solved:
                response, prompt_tokens, completion_tokens = self.run_single_pass(item)
                self.total_tokens += prompt_tokens + completion_tokens
                if hasattr(self, "parse_code"):
                    cur_imp = self.parse_code(response)
                else:
                    cur_imp = parse_response(response)

                item["source_codes"].append(cur_imp)
                item["responses"].append(response)
                item["prompt_tokens"].append(prompt_tokens)
                item["completion_tokens"].append(completion_tokens)
                item["no_of_try"] += 1

                is_solved = self.data.evaluate(
                    item=item,
                    cur_imp=cur_imp,
                    language=self.language
                )

                cur_pass += 1

            if is_solved:
                num_success += 1

            item["is_solved"] = is_solved
            item["language"] = self.language
            item["task_id"] = item[self.data.id_key]


            if i < len(self.results):
                self.results.results[i] = item
                self.results.save_results()
            else:
                self.results.add_result(item)

            if self.verbose:
                print(
                    f'completed {i+1}/{num_items}, Solved: {self.results[i]["is_solved"]}, number of success = {num_success}/{i+1}, acc = {round(num_success/(i+1)*100, 2)}')
            temp = f"number of success = {num_success}/{i+1}, acc = {round(num_success/(i+1)*100, 2)}"
            
        with open(self.data.get_path(self.path), 'a') as f:
            f.write(f"{self.name}{temp} total_tokens = {self.total_tokens}\n")
        return True
