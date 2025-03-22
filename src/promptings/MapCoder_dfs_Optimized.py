from typing import List
import tiktoken
import os
import json
import re
import sys
import time
from promptings.agents import *
from promptings.node import *
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
        max_depth = 3,
        max_iter = 10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t
        self.sample_io_prompt: str
        self.cur_code: str
        self.max_depth = max_depth
        self.max_iter = max_iter
        
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
    
    def run_single_pass(self, item:dict):
        task = self.data.get_prompt(item)
        agent = Agent("python", task)
        root = Node("")
        root.set_depth(-1)
        
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"
        agent.set_sample_io_prompt(sample_io_prompt)
        
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "# Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""
        agent.set_std_input_prompt(std_input_prompt)
        
       # planning
        planning = agent.planning_dfs_optimized(self.k)
        agent.get_input("planning", planning)
        
        plans, pr_tok, com_tok = self.gpt_chat(
            planning
        )
        
        item['api_calls'] = item.get('api_calls', 0) + 1
        
        agent.get_output("planning ", plans)
        
        plans = self.replace_tag(plans, 'description')
        plans = self.replace_tag(plans, 'reasoning')
        plans = self.parse_xml(plans)
        
        # planning verification
        for plan in plans['plan']:
            planning_verification_prompt = agent.planning_verification_optimized(plan['description'])
            agent.get_input("planning verification", planning_verification_prompt)
            
            verified_plan, pr_tok_1, com_tok_1 = self.gpt_chat(
                planning_verification_prompt
            )
            
            agent.get_output("planning verification", verified_plan)
            
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1 
            
            verified_plan = self.replace_tag(verified_plan, 'reasoning')
            verified_plan = self.replace_tag(verified_plan, 'score')
            verified_plan = self.parse_xml(verified_plan)
            
            new_Node = Node(plan, int(verified_plan['score']))
            new_Node.set_depth(root.depth+1)
            root.children.append(new_Node)
        
        root.sort_children()
        
        passed = False
        stack = [root]
        iter = 0
        while stack and not passed and iter < self.max_iter:
            node = stack.pop()

            if node.depth >= self.max_depth:
                continue
            print("--------------\n\n")
            print("Cur depth ", node.depth)
            print("cur iter ", iter)
            print("\n\n--------------")
            for child in node.children:
                child.set_depth(node.depth+1)
            
                if iter > self.max_iter: break
                
                iter += 1
                
                coding_prompt = agent.coding_optimized(child.plan)
                code, pr_tok_1, com_tok_1 = self.gpt_chat(
                    coding_prompt
                )            
                
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                code = self.parse_code(code)
                agent.get_output("coding agent", code)
                
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )
                
                if passed:
                    return code, pr_tok, com_tok

                # debugging
                
                reflection_prompt = agent.reflection_optimized(self.k, test_log, code)
                agent.get_input("reflection_agent", reflection_prompt)
                
                reflection, pr_tok_1, com_tok_1 = self.gpt_chat(
                    reflection_prompt
                )
                
                agent.get_output("reflection_agent", reflection)
                
                reflection = self.replace_tag(reflection,"description")
                reflection = self.replace_tag(reflection,"reasoning")
                reflection = self.parse_xml(reflection)
                
                for relfected_plan in reflection['plan']:
                    planning_verification_prompt = agent.planning_verification_optimized(relfected_plan['description'])
                    agent.get_input("reflection verification", planning_verification_prompt)
                    
                    verified_reflection, pr_tok_1, com_tok_1 = self.gpt_chat(
                        planning_verification_prompt
                    )

                    agent.get_output("reflection verification", verified_reflection)
                    item['api_calls'] += 1
                    pr_tok += pr_tok_1
                    com_tok += com_tok_1 

                    verified_reflection = self.replace_tag(verified_reflection, "reasoning")
                    verified_reflection = self.replace_tag(verified_reflection, "score")
                    verified_reflection = self.parse_xml(verified_reflection)
                    
                    score = int(verified_reflection['score'])
                    if score > node.score:
                        new_Node = Node(plan, score)
                        new_Node.set_depth(node.depth+1)
                        child.children.append(new_Node)
                
                child.sort_children()
                stack.append(child)
                
        return code, pr_tok, com_tok