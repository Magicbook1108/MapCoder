class Agent:
    def __init__(self, language, task):
        self.language = language
        self.task = task
        self.algorithm_prompt = None
        self.std_input_prompt = None
        self.sample_io_prompt = None

    def set_algorithm_prompt(self, algorithm_prompt):
        self.algorithm_prompt = algorithm_prompt

    def set_std_input_prompt(self, std_input_prompt):
        self.std_input_prompt = std_input_prompt
    
    def set_sample_io_prompt(self, sample_io_prompt):
        self.sample_io_prompt = sample_io_prompt
    
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
            "content": f"""
Given a competitive programming problem and a set of plan to solve the problem in {self.language},
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
            "content": f"""
Generate exactly {k} distinct plans to solve this competitive programming problem. Follow the format strictly.

# Problem:
{self.task}
{sample_io_prompt}

Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Your response must follow the following xml format -
For each plan, you must have a pair of 
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
    def planning_dfs(self, k):
                return [
        {
                "role": "user",
                "content": f"""
Given a competitive programming problem generate a concrete planning to solve the problem.
Generate exactly {k} distinct plans to solve this competitive programming problem. 

## Problem to be solved:
{self.task}
{self.sample_io_prompt}

Important:
Do NOT use `<`, `>`, `&` symbols directly in your planning description.
Do NOT leave an unclosed <description> or a missing opening tag.
You should give only the planning to solve the problem. Do not add extra explanation or words.

## Planning:
Each plan must have a plan id

""",
            },
        ]
    
    def planning_verificaion_dfs(self, planning):
        return [
            {
                "role": "user",
                "content": f"""
    Given a competitive programming problem and a set of plans to solve the problem in {self.language},
    evaluate whether each plan is correct and feasible.

    # Problem:
    {self.task}

    # Planning:
    {planning}

    ----------------
    Important:
    Each plan must be enclosed in a separate `<plan></plan>` tag. **Do NOT put multiple plans inside one `<description></description>`**.
    Your response must follow the XML format below:

    ```xml
    <root>
    <plan>
    <description>
    Copy the original planning here.
    </description>
    <confidence>
    Confidence score (0-100) indicating how well the plan can solve the problem.
    </confidence>
    </plan>
        
        <!-- Add more plans here if needed -->
    </root>
    """}]
    
    def reflection_agent(self, k, sample_io_prompt, test_log, code):
        return [
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
You should give only the planning to solve the problem. Do not add extra explanation or words.

## Planning:
Each plan must have a plan id

""",
        }
        ]
    
    # dfs_optimized
    def planning_dfs_optimized(self, k):
                return [
        {
                "role": "user",
                "content": f"""
Given a programming task and a sample input-output prompt, generate {k} distinct plans outlining different strategies for implementing the solution.
Provide a structured reasoning for each plan that explains the thought process behind the approach. Ensure that the plans are actionable and clearly delineate the steps necessary to solve the task.

## Problem to be solved:
{self.task}
{self.sample_io_prompt}

Your response must follow the XML format, make sure each opening tag `<tag>` has a corresponding closing tag `</tag>`:

<root>
<plan>
<reasoning>
Reasoning here
</reasoning>
<description>
Planning to solve the problem, each plan is considered as a single string contain multiple steps
</description>
</plan>

Add more plans here
</root>
""",
            },
        ]
                
    def planning_verification_optimized(self, planning):
        return [
            {
                "role": "user",
                "content": f"""
Evaluate the given plan by generating a detailed reasoning that explains the thought process behind the plan's execution.
And assign a score that quantifies its feasibility and effectiveness based on the criteria established for successful task completion.

# Problem:
{self.task}

# Planning:
{planning}

Your response must follow the XML format, make sure each opening tag `<tag>` has a corresponding closing tag `</tag>`:

<root>
<reasoning>
Reasoning here
</reasoning>
<score>
Score here, a single str from 0-100
</score>
</root>
"""}]
    
    def coding_optimized(self, planning):
        return [
            {
        "role": "user",
        "content":
            f"""In a high-stakes programming competition, you are tasked with transforming a given problem into executable code.
Given the fields task, plan, and sample_io_prompt, your objective is to produce {self.language} code.
Remember, the accuracy of your solution could determine the outcome of the competition, so be thorough and precise in your coding.

## Problem to be solved:
{self.task}
{self.sample_io_prompt}

## Planning:
{planning}


----------------
Important:
{self.std_input_prompt}

# Your response must contain only the {self.language} code to solve this problem. 
Do not add extra explanation or words.
"""
   }
]
    
    def reflection_optimized(self, k, test_log, code):
        return [
        {
            "role":"user",
            "content":f"""
Given the task description, and the test log from previous attempts, generate {k} distinct plans for solving the programming task.
Ensure to provide reasoning for each plan to clarify the decision-making process behind the proposed approaches.
Utilize the sample input/output prompt to guide the planning process effectively.

## Problem to be solved:
{self.task}

{self.sample_io_prompt}

# Test log
{test_log}

# Code
{code}

Your response must follow the XML format, make sure each opening tag `<tag>` has a corresponding closing tag `</tag>`:

<root>
<plan>
<reasoning>
Reasoning here
</reasoning>
<description>
Planning to solve the problem, each plan is considered as a single string contain multiple steps
</description>
</plan>

Add more plans here
</root>

""",
        }
        ]