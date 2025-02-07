from typing import List
import tiktoken
import os
import json
import re
import sys
import time

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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t

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


    def knowledge_agent(self,problem):
        
        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{problem}

# Exemplars:
Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {self.language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in {self.language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>

# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
""",
            },
        ]

        print("\n\n________________________")
        print("Input for knowledge base and exemplars: ")
        print(input_kb_exemplars[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=input_kb_exemplars
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

        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response = self.parse_xml(response)
        
        return response, pr_tok, com_tok    
    
    def planning_agent(self, example, example_no, problem, algorithm_prompt, sample_io_prompt):
        
        example_problem = example["description"]
        example_planning = example["planning"]
        
        input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"""
Given a competitive programming problem generate a concrete planning to solve the problem.
\n# Problem:\n{example_problem}\n# Planning:\n{example_planning}\n{algorithm_prompt}
\n## Problem to be solved:\n{problem}\n{sample_io_prompt}
\n## Planning:\n\n----------------
\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words.
"""
                }
            ]

        print("\n\n________________________")
        print(
            f"Input for our problem planning using example: {example_no}: ")
        print(input_for_problem_planning[0]['content'], flush=True)       
        
        planning, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_problem_planning
        )          
        
        return planning, pr_tok_1, com_tok_1
    
    def planning_verification_agent(self, problem, planning):
        input_for_planning_verification = [
            {
                "role": "user",
                "content": f"""
Given a competitive programming problem and a plan to solve the problem in {self.language}, tell whether the plan is correct to solve this problem.

# Problem:
{problem}
# Planning:
{planning}

----------------
Important: Your response must follow the following xml format-```
<root>
<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>
```
"""
            }
        ]
        
        print("\n\n________________________")
        print("Input for planning verification: ")
        print(input_for_planning_verification[0]['content'], flush=True)

        verification_res, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_planning_verification
        )
        
        verification_res = self.replace_tag(
            verification_res, 'explanation')
        verification_res = self.replace_tag(verification_res, 'confidence')

        verification_res = self.parse_xml(verification_res)

        verification_res['confidence'] = int(
            str(verification_res['confidence']).strip())

        print("Response from planning verification: ")
        print(verification_res, flush=True)

        return verification_res['confidence'], pr_tok_1, com_tok_1
    
    
    def coding_agent(self,problem, planning, algorithm_prompt, sample_io_prompt, std_input_prompt):
                    
        
        input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"""
Given a competitive programming problem generate {self.language} code to solve the problem.
{algorithm_prompt}\n## Problem to be solved:\n{problem}\n## Planning:\n{planning}\n{sample_io_prompt}
## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}
## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
                }
            ]

        print("\n\n________________________")
        print("Input for final code generation: ")
        print(input_for_final_code_generation[0]['content'], flush=True)

        code, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_final_code_generation
        )        
    
        print("\n\n________________________")
        print("Response from final code generation: ")
        print(code, flush=True)

        return code, pr_tok_1, com_tok_1
            
    def code_improving_agent(self, problem, std_input_prompt, algorithm_prompt, response, test_log):
        
        input_for_improving_code = [
            {
                "role": "user",
                "content": f"""
Given a competitive programming problem you have generated {self.language} code to solve the problem. 
But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.
{algorithm_prompt}
## Problem to be solved:\n{problem}
{response}
## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.


----------------
Important:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem.
"""
            }
        ]        
        
        print("\n\n________________________")
        print("Input for improving code generation: ")
        print(input_for_improving_code[0]['content'], flush=True)
        
        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_improving_code
        )
        
        print("\n\n________________________")
        print("Response from improving code generation: ")
        print(response, flush=True)
        
        code = self.parse_code(response)
        
        return code, pr_tok_1, com_tok_1
    
    def task_distibute_agent(self, problem, sample_io_prompt, review_response):
        input_for_task_distributor = input_for_task_distributor = [
    {
        "role": "user",
        "content": f"""
        
Your task is to analyze the given problem and decompose it into smaller, well-defined sub-tasks. Each sub-task must:
- Solve a single logical step.
- Be independent from other sub-tasks as much as possible.
- Be implementable in {self.language} as a function.
- Include a sample input and output (following {sample_io_prompt}).

### Instructions:
1. Review the provided problem and the summarization of the previous failure.
2. Divide the problem into smaller sub-tasks:
   - Each sub-task must address a specific part of the problem.
   - Focus on avoiding issues highlighted in the previous failure.

3. For each sub-task:
   - Provide a concise description explaining its purpose and necessity.
   - Define the sample input and output for this task.

4. Construct a workflow:
   - Explain how the sub-tasks connect to form the complete solution.
   - Highlight how this workflow addresses the issues in the previous attempt.

### Summarization of Previous Failure:
{review_response}

### Problem:
{problem}

### Response Format:
Your response must strictly follow this XML structure:

<root>
  <problem>
  Write each problem in the following format.
    <description>
      # Explain the purpose of the sub-task, why it is necessary, and how it contributes to solving the problem.
    </description>
    <sample_io>
      # Generate three test case with input and output.
    </sample_io>
  </problem>
  
  <!-- Repeat the <problem> section for each sub-task -->

  <workflow>
    # Explain the overall workflow.
    # Describe how sub-tasks are connected, and how their dependencies are managed.
    # Ensure the workflow highlights how this approach avoids the issues from the previous failure.
  </workflow>
</root>

### Notes:
- Ensure the input to the first sub-task corresponds to the original problem input.
- The output of the last sub-task should match the original problem output.
- Your response must be clear, concise, and actionable.
"""
    }
]

        print("\n\n________________________")       
        print("Input for task distributor: ")
        print(input_for_task_distributor[0]['content'])
        
        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_task_distributor
        )
        
        response = self.trim_text(
            response, "# Define a sample input and output for this task." )
        response = self.trim_text(
            response, " Briefly describe the sub-task")
        response = self.replace_tag(response, "description")
        response = self.replace_tag(response, "sample_io")
        
        print("\n\n________________________")        
        print("Response from task distributor: ")
        print(response, flush=True)
        
        response = self.parse_xml(response)
        print("\n\n________________________")    
        print("parsed")
        print(response)
        return response, pr_tok_1, com_tok_1
    
    def reviewer_agent(self, code):
        input_for_reviewer = [
            {
                "role": "user",
                "content": f"""
    Analyze the given failure case and summarize why the previous attempt failed. Your report should include:
    1. **What went wrong**: Briefly describe the main reasons for the failure.
    2. **Issues found**: Highlight specific problems in the code or approach.
    3. **Suggestions**: Provide clear and actionable steps to help the next team improve and avoid similar mistakes.

    ### Previous Failure Case:
    {code}

    Keep your response concise and focused, but ensure it provides enough detail to guide the next team effectively.
    """
            }
        ]
        print("\n\n________________________")
        print("Input for reviewer:")
        print(input_for_reviewer[0]['content'])
        
        response, pr_tok_1, com_tok_1 = self.gpt_chat(input_for_reviewer)
        
        print("\n\n________________________")
        print("Response from reviewer:")
        print(response, flush=True)
        
        return response, pr_tok_1, com_tok_1
    
    def code_summarize_agent(self, code, workflow):
        input_for_summarize = [
    {
        "role": "user",
        "content": f"""
Your task is to combine the provided code snippets into a single, cohesive implementation following the given workflow.

### Input:
#### Code Snippets:
{code}

#### Workflow:
{workflow}

### Instructions:
1. Combine the code snippets in the order specified by the workflow.
2. Ensure that the combined code adheres to the workflow logic.
3. Return the final integrated code.

### Output:
Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
    }
]
        print("\n\n________________________")
        print("Input for code summarizer:")
        print(input_for_summarize[0]['content'])
        
        
        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_summarize
        )
        
        print("\n\n________________________")
        print("Response from code summarizer: ")
        print(response, flush=True)
        
        return response, pr_tok_1, com_tok_1
        
        
    def testcase_review_agent(self, code, sample_io_prompt):
        input_for_code_review = [
            {
    "role": "user",
    "content": f"""
Your task is to review the given code and test cases to determine if the code passes all test cases. Provide a structured XML report.

### Code:
{code}

### Test Cases:
{sample_io_prompt}

### Instructions:
1. Execute the code using the provided test cases.
2. For each test case:
   - Show the input, expected output, and actual output.
   - Determine whether the result is correct.
3. If any test fails:
   - Identify the failing test case.
   - Explain why the failure occurred.
4. Summarize whether all tests passed or not.

----------------
Important:
Your response must follow the following xml format-

<root>
  <passed>
    # TRUE if all tests passed, FALSE otherwise.
  </passed>
  <problem>
    # If any test failed, explain why it failed. Leave empty if all tests passed.
  </problem>
</root>

### Notes:
- Ensure your analysis is clear and concise.
- Follow the XML format strictly to maintain consistency.
"""
}
        ]

        print("\n\n________________________")      
        print("Input for test case review:")
        print(input_for_code_review[0]['content'], flush=True)
        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_code_review
        )      
        
        response = self.replace_tag(response, "passed")
        response = self.replace_tag(response, "problem")
        
        print("\n\n________________________")  
        print("Response from test case review")
        print(response, flush=True)      
        
        response = self.parse_xml(response)
        
        return response, pr_tok_1, com_tok_1
        
        
    def run_pass(self,item:dict):
        
        problem = self.data.get_prompt(item)
        
        response, pr_tok, com_tok = self.knowledge_agent(problem)
        item['api_alls'] = item.get('api_calls',0) + 1
        
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{ response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(item['sample_io'])}\n"
        
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""
    
        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            
            planning, pr_tok_1, com_tok_1 = self.planning_agent(example, example_no, problem, algorithm_prompt, sample_io_prompt)
            item['api_alls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            
            confidence, pr_tok_1, com_tok_1 = self.planning_verification_agent(problem, planning)
            item['api_alls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            
            plannings.append((
                planning,
                confidence,
                example
            ))
            
        plannings.sort(key=lambda x: x[1], reverse=True)
        

        for planning_with_ex in plannings:
            planning, confidence, example = planning_with_ex
            
            code, pr_tok_1, com_tok_1 = self.coding_agent(problem, planning, algorithm_prompt, sample_io_prompt,std_input_prompt)
            item['api_alls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            
            response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            
            passed = False
            for i in range(1, self.t+1):
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    code,
                    self.language
                )
                
                if passed:
                    break
            
                print(f"Input for improving code generation: {i}")
                
                code, pr_tok_1, com_tok_1 = self.code_improving_agent(problem, std_input_prompt, algorithm_prompt, response, test_log)
                item['api_alls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                
            if passed:
                return code, pr_tok, com_tok
            
        review_response, pr_tok_1, com_tok_1 = self.reviewer_agent(code)
        item['api_alls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1
        
        response, pr_tok_1, com_tok_1 = self.task_distibute_agent(problem, sample_io_prompt, review_response)
        item['api_alls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        workflow = response['workflow']

        task_solution = []
        for task_no, task in enumerate(response['problem'], start=1):
            sample_io_prompt = f"## Sample Test cases: \n{task['sample_io']}\n"
            
            problem = task['description']
            
            response, pr_tok_1, com_tok_1 = self.knowledge_agent(problem)
            item['api_alls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            
            algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{ response['algorithm']}"
            
            plannings = []
            
            for example_no, example in enumerate(response["problem"], start=1):
                planning, pr_tok_1, com_tok_1 = self.planning_agent(example, example_no, problem, algorithm_prompt, sample_io_prompt)
                item['api_alls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                
                confidence, pr_tok_1, com_tok_1 = self.planning_verification_agent(problem, planning)
                item['api_alls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1           
            
                plannings.append((
                    planning,
                    confidence,
                    example
                ))
                
            plannings.sort(key=lambda x: x[1], reverse=True)
            
            
            for planning_with_ex in plannings:
                planning, confidence, example = planning_with_ex
            
                code, pr_tok_1, com_tok_1 = self.coding_agent(problem, planning, algorithm_prompt, sample_io_prompt,std_input_prompt)
                item['api_alls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                
                passed = False
                for i in range(1, self.t+1):
                    
                    response, pr_tok_1, com_tok_1 = self.testcase_review_agent(code,sample_io_prompt)
                    item['api_alls'] += 1
                    pr_tok += pr_tok_1
                    com_tok += com_tok_1
                    
                    
                    if(response['passed'] == "TRUE"):
                        passed = True
                        task_solution.append((task_no, code))
                        break
                    
                    test_log = response['problem']
                    
                    print(f"Input for improving code generation: {i}")
                    code, pr_tok_1, com_tok_1 = self.code_improving_agent(problem, std_input_prompt, algorithm_prompt, response, test_log)
                    item['api_alls'] += 1
                    pr_tok += pr_tok_1
                    com_tok += com_tok_1                    
                
                if passed:
                    break
                
        task_solution_str = "\n".join(
        [
        f"""
        Task {task_no + 1}:
        {solution}
        ________________________
        """
        for task_no, solution in enumerate(task_solution)
        ]
        )
                    
        code, pr_tok_1, com_tok_1 = self.code_summarize_agent(task_solution_str, workflow)
        item['api_alls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1
        
        passed, test_log = self.data.evaluate_sample_io(
            item,
            code,
            self.language
        )
        
        return code, pr_tok, com_tok
        