# prompts.py

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


def get_complexity_agent(task):
    """
    生成动态评估任务复杂度的提示内容，返回复杂度评分（1-10）
    响应严格为单个整数（无其他文本）
    """
    return [
        {
            "role": "user",
            "content": f"""Analyze the complexity of the given programming task and strictly respond with a single integer from 1 to 10.

# Complexity Criteria:
1 - Trivial: Simple operations (e.g., printing output, basic arithmetic)
2 - Very Easy: Basic loops and conditionals (e.g., sum calculation, string reversal)
3 - Easy: Requires simple brute-force algorithms (e.g., nested loops, linear search, basic recursion)
4 - Lower-Mid: Requires standard sorting or binary search (e.g., merge sort, quicksort, basic sorting problems)
5 - Moderate: Requires advanced sorting techniques or graph traversal (e.g., BFS/DFS, backtracking, segment tree basics)
6 - Upper-Mid: Needs efficient algorithms and custom sorting (e.g., lexicographical sorting with constraints, Dijkstra's algorithm)
7 - Hard: Involves complex data structures like Trie, Fenwick Tree, advanced DP (e.g., suffix array, topological sorting)
8 - Very Hard: Requires multi-step optimizations and computational geometry (e.g., max flow, Heavy-Light Decomposition)
9 - Expert: Uses rare algorithms such as Mo’s Algorithm, Suffix Automaton, and numerical optimization
10 - Legendary: Research-level complexity (e.g., Quantum Algorithms, AI/ML-driven combinatorial optimization)

Task Description:
{task}

Your response must be a single integer from 1 to 10 with no additional explanation.
"""
        }
    ]


def get_knowledge_retrieval_agent(k, problem):
    complexity = ""
    if k >= 5:
        complexity = "This problem requires advanced techniques and cannot be solved with naive or brute-force methods."
    
    double_k = 2 * k
    """
    根据给定的 problem 和其他参数，生成“知识库+示例”的提示内容。
    """
    return [
        {
            "role": "user",
            "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{problem}

# Exemplars:
Recall {double_k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate python code step by step to solve that problem
3. finally generate a planning to solve that problem



# Algorithm:

----------------
Important:
Your response must follow the following format, remember to remove the line start with # and end with ^ with output.
{complexity}

Problem id:
# Recall {double_k} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format. ^
# Give the id start from 1 ^

Description:
# Describe the problem.^
End Description 

Code:
# Let's think step by step to solve this problem in python programming language ^
End code

Planning:
# Planning to solve this problem. ^
End planning


# similarly add more problems here... ^

"""
        }
    ]
    
def get_k_problem_agent(k, problem, double_k_problems):
    """
    选择与给定 problem 最相似的 k 个问题，并转换为 XML 格式。
    """
    return [
        {
            "role": "user",
            "content": f"""Given a list of generated problems, select the top {k} most similar problems to the given problem.
        
# Problem:
{problem}

# Instructions:
1. You must strictly select the {k} most similar problems from the provided generated problems.
2. Do NOT modify the content of the selected problems. Keep their descriptions, code, and planning exactly as they are.
3. Ensure the selected problems are distinct and do not repeat the original problem.

# Generated Problems:
{double_k_problems}

----------------
# Output Format:
Your response must follow the following xml format-
Ensure that every opening tag `<tag>` has a corresponding closing tag `</tag>`. No missing or extra tags are allowed.

<root>
<problem>
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in python programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>

# Add remainning problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high-level generic tutorial for solving these types of problems. Do not generate code.
</algorithm>
</root>
"""
        }
    ]



def get_planning_agent(example_problem, example_planning, algorithm_prompt, task, sample_io_prompt):
    """
    生成“我们的实际问题规划”需要的提示内容。
    """
    return [
        {
            "role": "user",
            "content": f"""
Given a competitive programming problem generate a concrete planning to solve the problem.
# Problem:
{example_problem}
# Planning:
{example_planning}

{algorithm_prompt}

## Problem to be solved:
{task}
{sample_io_prompt}

## Planning:

----------------
Important: You should give only the planning to solve the problem. Do not add extra explanation or words.
"""
        }
    ]


def get_planning_verification(problem, planning):
    """
    生成“规划验证”需要的提示内容。
    """
    return [
        {
            "role": "user",
            "content": f"""Evaluate the structural validity of the proposed problem-solving plan (NOT its code implementation) based on the criteria below.

# Problem:
{problem}

# Proposed Plan:
{planning}

# Scoring Criteria:
1 - Fundamentally Incorrect: The approach is fundamentally flawed (e.g., using sorting instead of searching).
2 - Incomplete Plan: Addresses core logic but misses at least two critical steps (e.g., lacks input validation or key sub-procedures).
3 - Basic but Flawed: Covers the main approach but omits one crucial component (e.g., does not account for edge cases).
4 - Nearly Complete: Covers all essential steps with only minor gaps (e.g., unspecified iteration order, missing optimization).
5 - Structurally Sound and Comprehensive: The plan is logically well-structured and covers:
   - Correct algorithm selection
   - Preprocessing/post-processing steps
   - Handling of edge cases
   - Consideration of computational complexity

Important:
Your response must follow the following format, remember to remove the line start with # and end with ^ with output.

Score:
# Must be an integer between 1-5 ^

Explanation:
# Explain how the plan meets or fails to meet the structural requirements. ^
"""
        }
    ]

def get_final_planning_agent(plannings):
    return [
        {
            "role":"user",
            "content":f"""
Given a set of plans, filter and merge them according to the following rules.

# Exemplars:
1. Remove all plans with a score of 1.
2. Keep all plans with a score of 5.
3. For plans with scores between 2 and 4:
   - Identify the two most similar plans and pair them together.
   - If the number of plans is odd, leave the least similar one unmerged.
   - In each pair, prioritize the higher-scoring plan as the main structure.
   - The lower-scoring plan should provide supplementary improvements.
   - If two paired plans have the same score, both should be retained separately without merging.
   - After merging a pair, remove them from further consideration and continue pairing the next most similar plans.

# Plannings:
{plannings}

----------------
Your response must strictly follow the XML format below, remember to remove the line start with # and end with ^ with output.
Ensure that every opening tag `<tag>` has a corresponding closing tag `</tag>`. No missing or extra tags are allowed.
You must arrange them from higher to lower score. The ones after merging should retain the original score.

<root>
<Plan>
<score>
# Must be an integer between 1-5 ^
</score>
<description>
# The original plan or the improved plan after merging ^
</description>
<Plan>

# Add remainning plans here...

</root>
            """
        }
    ]

def get_coding_agent(algorithm_prompt, task, planning, sample_io_prompt, std_input_prompt):
    """
    生成最终代码需要的提示内容。
    """
    return [
        {
            "role": "user",
            "content": f"""Given a competitive programming problem generate python code to solve the problem.
## Relevant Algorithm to solve the next problem:
{algorithm_prompt}

## Problem to be solved:
{task}

## Planning:
{planning}

## Sample Test cases:
{sample_io_prompt}

## Let's think step by step.

----------------
Important:
{std_input_prompt}
## Your response must contain only the python code to solve this problem. Do not add extra explanation or words.
"""
        }
    ]


def get_code_debug(algorithm_prompt, task, response, test_log, std_input_prompt):
    """
    生成改进代码需要的提示内容。
    """
    return [
        {
            "role": "user",
            "content": f"""Given a competitive programming problem you have generated python code to solve the problem. 
But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.

## Relevant Algorithm to solve the next problem:
{algorithm_prompt}

## Problem to be solved:
{task}

{response}

## Test Report:
{test_log}

## Modified Planning:
## Let's think step by step to modify python Code for solving this problem.

----------------
Important:
{std_input_prompt}
## Your response must contain the modified planning and then the python code inside ``` block to solve this problem.
"""
        }
    ]

def get_input(agent_name, prompt):
    print("\n\n________________________")
    print(f"Input for {agent_name}: ")
    print(prompt[0]['content'],flush=True)

def get_output(agent_name, response):
    print("\n\n________________________")
    print(f"Response from {agent_name}: ")
    print(response, flush=True)