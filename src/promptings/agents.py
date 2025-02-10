class Agent:
    def __init__(self, language, task):
        self.language = language
        self.task = task
        self.algorithm_prompt = None
        self.std_input_prompt = None

    def set_algorithm_prompt(self, algorithm_prompt):
        self.algorithm_prompt = algorithm_prompt

    def set_std_input_prompt(self, std_input_prompt):
        self.std_input_prompt = std_input_prompt
    
    def knowledge_retrieval_agent_prompt(self,k):
        return [
            {
                "role": "user",
                "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{self.task}

# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {self.language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {k} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
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
    
    def planning_agent_prompt(self, example_problem, example_planning, sample_io_prompt):
        return [
        {
                "role": "user",
                "content": f"""Given a competitive programming problem generate a concrete planning to solve the problem.


# Problem:
{example_problem}

# Planning:
{example_planning}
{self.algorithm_prompt}

## Problem to be solved:
{self.task}
{sample_io_prompt}

Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Do NOT leave an unclosed <description> or a missing opening tag.
You should give only the planning to solve the problem. Do not add extra explanation or words.
## Planning:

""",
            },
        ]
    
    def planning_verification_agent_prompt(self, planning):
        return [
                {
                    "role": "user",
                    "content": f"""Given a competitive programming problem and a plan to solve the problem in {self.language}, tell whether the plan is correct to solve this problem.

# Problem:
{self.task}
# Planning:
{planning}\
    
----------------
Important: Your response must follow the following xml format-```
<root>
<explanation> 
Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>
<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>
</root>
```"""
                }
            ]

    def coding_agent_prompt(self, planning, sample_io_prompt):
        return [
            {
        "role": "user",
        "content":
            f"""Given a competitive programming problem, generate {self.language} code to solve the problem.
{self.algorithm_prompt}
## Problem to be solved:
{self.task}
## Planning:
{planning}
{sample_io_prompt}
## Let's think step by step.
----------------
Important:
{self.std_input_prompt}
## Your response must contain only the {self.language} code to solve this problem. 
Do not add extra explanation or words.
"""
   }
]
    
    def debugging_agent_prompt(self, response, test_log):
        return [
    {
        "role": "user",
        "content": f"""Given a competitive programming problem, you have generated {self.language} code to solve the problem. But the generated code cannot pass sample test cases. Improve your code to solve the problem correctly.

{self.algorithm_prompt}

## Problem to be solved:
{self.task}

{response}

## Test Report:
{test_log}

## Modified Planning:
## Let's think step by step to modify the {self.language} code for solving this problem.

----------------
Important:
{self.std_input_prompt}

## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."""
    }
]

    def get_input(self, agent_name, prompt):
        print("\n\n________________________")
        print(f"Input for {agent_name}: ")
        print(prompt[0]['content'],flush=True)

    def get_output(self, agent_name, response):
        print("\n\n________________________")
        print(f"Response from {agent_name}: ")
        print(response, flush=True)

    
    # Merge plans into a single
    def merge_single_planning_agent_prompt(self, plans):
        return [
    {
        "role": "user",
        "content":f"""
Merge the provided plans into a single unified plan. 
Prioritize elements from plans with higher confidence scores. 
Ensure the final plan is coherent and incorporates the most reliable information.


Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Do NOT leave an unclosed <description> or a missing opening tag.
You should give only the planning to solve the problem. Do not add extra explanation or words


## Plannings:
{plans}

## Plan:

        """
    }
]
    def merge_planning_verification_agent_prompt(self, planning):
        return [
            {
            "role": "user",
            "content": f"""Given a competitive programming problem and a set of plan to solve the problem in {self.language},
tell whether the plan is correct to solve this problem.

# Problem:
{self.task}

# Planning:
{planning}
----------------
You must not change the planning, only provide a score between 1 - 100.

# Output Format:
# Planning
[Copy the original planning here]

# Score:
[Your score between 1 and 100]
"""}
        ]
    
    # Mapcoer without knowledge retrieval
    def planning_without_knowledge_retrieval_prompt(self, k, sample_io_prompt):
        return [
        {
            "role": "user",
            "content": f"""Generate exactly {k} distinct plans to solve this competitive programming problem. Follow the format strictly.

# Problem:
{self.task}
{sample_io_prompt}

Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Do NOT leave an unclosed <description> or a missing opening tag.
Your response must follow the following xml format -

<root>
<problem>
# Recall {k} plans to solve the problem
<description>
# Describe the plan here
</description>
</problem>

# similarly add more plans here...
</root>

"""
        }
    ]

    # dfs
    def reflection_agent(self, k, sample_io_prompt, test_log, code):
        {
            "role":"user",
            "content":f"""
Given a competitive programming problem, you have generated {self.language} code to solve the problem. 
But the generated code cannot pass sample test cases. Generate exactly {k} distinct plans to solve this competitive programming problem. 
You should first identify the problem in the given solution, then explains how to solve this issues.

# Problem:
{self.task}
{sample_io_prompt}

# Test log
{test_log}

# Code
{code}

Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Do NOT leave an unclosed <description> or a missing opening tag.
Your response must follow the following xml format -

<root>
<problem>
# Recall {k} plans to solve the problem
<description>
# Describe the plan here
</description>
</problem>

# similarly add more plans here...
</root>

"""
        }