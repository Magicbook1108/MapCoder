from typing import List
import tiktoken
import os
import json
import re
import sys
import time
import promptings.agents as agents
from copy import deepcopy
import xml.etree.ElementTree as ET

from .Base import BaseStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from results.Results import Results
from evaluations.func_evaluate import evaluate_io

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

# KB + Exemplars + Example Planning + Problem Planning + Code Generation + Sample IO testing + Code Improvement


class MapCoder(BaseStrategy):
    def __init__(
        self,
        k: int = 5,
        t: int = 5,
        usage: dict = {},
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t
        self.usage = usage

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return self.xml_to_dict(root)

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str
    
    def update_usage(self,item:dict,pr_tok,com_tok) -> None:
        item.setdefault('api_calls', 0)
        self.usage.setdefault('pr_tok', 0)
        self.usage.setdefault('com_tok', 0)
        self.usage.setdefault('api_calls', 0)

        # 更新计数
        item['api_calls'] += 1
        self.usage['pr_tok'] += pr_tok
        self.usage['com_tok'] += com_tok
        self.usage['api_calls'] = item['api_calls']
        
    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

    def run_single_pass(self, item: dict):
        print("", flush=True)

        task = self.data.get_prompt(item)
        
        # Give complexity scores (which is K) for each task
        complexity_name = "Complexity agent"
        complexity_prompt = agents.get_complexity_agent(task)
        
        agents.get_input(complexity_name, complexity_prompt)
        
        response, pr_tok, com_tok = self.gpt_chat(
            processed_input= complexity_prompt
        )
        
        # The amount of similar problems we finally need
        k = 2 if int(response) < 2 else int(response)
        
        self.update_usage(item, pr_tok, com_tok)
        
        agents.get_output(complexity_name, response)
        
        # Knowledge Retrieval Agent, generate 2k problems
        kb_name = "knowledge retrieval agent"
        kb_prompt = agents.get_knowledge_retrieval_agent(k, task)

        agents.get_input(kb_name, kb_prompt)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input= kb_prompt
        )

        self.update_usage(item,pr_tok,com_tok)
        
        agents.get_output(kb_name, response)
        
        # Select top k ones
        k_problem_name = "get k problem agent"
        k_problem_prompt = agents.get_k_problem_agent(k, task, response)
        
        agents.get_input(k_problem_name, k_problem_prompt)
        
        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=k_problem_prompt
        )
        
        # Post processing
        response = self.trim_text(
            response, "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response, "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
        response = self.trim_text(
            response, "# Planning to solve this problem:")
        response = self.trim_text(
            response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')
        
        self.update_usage(item,pr_tok,com_tok)
        
        agents.get_output(k_problem_name, response)
        
        response = self.parse_xml(response)
            
        print(response)
        
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{ response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"
        # if type(self.data) != MBPPDataset and type(self.data) != XCodeDataset else ""

        plannings = []
        plannings_str = ""
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

            planning_name = "planning agent"
            planning_prompt = agents.get_planning_agent(example_problem, example_planning, algorithm_prompt, task, sample_io_prompt)
            
            print(f"Input for our problem planning using example: {example_no}: ")
            agents.get_input(planning_name, planning_prompt)

            planning, pr_tok, com_tok = self.gpt_chat(
                planning_prompt
            )

            self.update_usage(item, pr_tok, com_tok)

            agents.get_output(planning_name, planning)

            
            planning_verification_name = "planning verification agent"
            plannign_verification_prompt = agents.get_planning_verification(task, planning)

            agents.get_input(planning_verification_name, plannign_verification_prompt)
            
            verification_res, pr_tok, com_tok = self.gpt_chat(
                processed_input=plannign_verification_prompt
            )

            self.update_usage(item, pr_tok, com_tok)

            plannings_str += f"Plan id: {example_no} \n {verification_res} \n\n"
        
        final_planning_name = "final planning agent"
        final_planning_prompt = agents.get_final_planning_agent(plannings_str)
        
        agents.get_input(final_planning_name, final_planning_prompt)
        
        plannings, pr_tok, com_tok = self.gpt_chat(
            processed_input= final_planning_prompt
        )
        
        plannings = self.replace_tag(plannings, 'description')
        plannings = self.replace_tag(plannings, 'score')
        
        self.update_usage(item,pr_tok, com_tok)
        
        agents.get_output(final_planning_name, plannings)


        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        for plan_no, plan in enumerate(plannings["plan"], start = 1):

            coding_agent_name = "coding agent"
            coding_agent_prompt = agents.get_coding_agent(algorithm_prompt, task, planning, sample_io_prompt, std_input_prompt)

            agents.get_input(coding_agent_name, coding_agent_prompt)

            code, pr_tok, com_tok = self.gpt_chat(
                coding_agent_prompt
            )

            self.update_usage(item, pr_tok, com_tok)
            
            code = self.parse_code(code)
            
            agents.get_output(coding_agent_name, code)
            
            response = f"## Planning: {plan}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )

                if passed:
                    break

                print(f"Input for improving code generation: {i}")

                debugging_agent_name = "debugging agent"
                debugging_agent_prompt = agents.get_code_debug(algorithm_prompt, task, response, test_log, std_input_prompt)

                agents.get_input(debugging_agent_name, debugging_agent_prompt)

                response, pr_tok, com_tok = self.gpt_chat(
                    debugging_agent_prompt
                )
                code = self.parse_code(response)
                
                self.update_usage(item, pr_tok, com_tok)

                agents.get_output(debugging_agent_name, response)

            # got a code that passed all sample test cases
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok