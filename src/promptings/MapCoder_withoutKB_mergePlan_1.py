# No KB, but still merge
from typing import List
import tiktoken
import os
import json
import re
import sys
import time
from promptings.agents import *
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
        k: int = 3,
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
    
    def get_input(agent_name, prompt):
        print("\n\n________________________")
        print(f"Input for {agent_name}: ")
        print(prompt[0]['content'],flush=True)

    def get_output(agent_name, response):
        print("\n\n________________________")
        print(f"Response from {agent_name}: ")
        print(response, flush=True)

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

        agents = Agent("python",task)

        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"

        planning_name = "planning agent"
        planning_prompt = agents.planning_without_knowledge_retrieval_prompt(self.k, sample_io_prompt)

        agents.get_input(planning_name, planning_prompt)

        plannings, pr_tok, com_tok = self.gpt_chat(
            planning_prompt
        )

        item['api_calls'] = item.get('api_calls', 0) + 1

        plannings = self.replace_tag(plannings, 'plan')

        agents.get_output(planning_name, plannings)

        plannings = self.parse_xml(plannings)

        plannings_with_score = ""
        for plan_no, problem in enumerate(plannings["problem"], start=1):
            plan = problem['description']

            planning_verification_name = "planning verification agent"
            plannign_verification_prompt = agents.planning_verification_agent_prompt(plan)

            agents.get_input(planning_verification_name, plannign_verification_prompt)
        
            verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
                plannign_verification_prompt
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            verification_res = self.replace_tag(
                verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')

            verification_res = self.parse_xml(verification_res)

            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())

            agents.get_output(planning_verification_name, verification_res)

            plannings_with_score += f"plan id: {plan_no} \n {plan} \n score: {verification_res[confidence]} \n\n"

        merge_single_planning_name = "Merge single planning"
        merge_single_planning_prompt = agents.merge_single_planning_agent_prompt(plannings_with_score)

        agents.get_input(merge_single_planning_name, merge_single_planning_prompt)

        merged_plan, pr_tok_1, com_tok_1 = self.gpt_chat(
            merge_single_planning_prompt
        )

        agents.get_output(merge_single_planning_name, plan)

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        agents.set_std_input_prompt(std_input_prompt)

        coding_agent_name = "coding agent"
        coding_agent_prompt = agents.coding_agent_prompt(merged_plan, sample_io_prompt)

        agents.get_input(coding_agent_name, coding_agent_prompt)

        code, pr_tok_1, com_tok_1 = self.gpt_chat(
            coding_agent_prompt
        )

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        code = self.parse_code(code)

        agents.get_output(coding_agent_name, code)

        response = f"## Planning: {merged_plan}\n## Code:\n```\n{code}\n```"
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
            debugging_agent_prompt = agents.debugging_agent_prompt(response, test_log)

            agents.get_input(debugging_agent_name, debugging_agent_prompt)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                debugging_agent_prompt
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            agents.get_output(debugging_agent_name, response)

            code = self.parse_code(response)


        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok