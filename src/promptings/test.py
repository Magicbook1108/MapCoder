import xml.etree.ElementTree as ET
import re

def fix_malformed_xml(response):
    """
    自动修复 `description` 和 `CDATA` 相关的格式错误
    """
    response = re.sub(r'<description>\s*([^<]*?)\s*\]\]></description>', 
                      r'<description><![CDATA[\1]]></description>', 
                      response)
    return response

def clean_code_blocks(response):
    """
    清理 `code` 部分的 Markdown 语言声明，例如 `python`
    """
    response = re.sub(r'<code><!\[CDATA\[\s*(python|cpp|java|c\+\+|javascript)\s*', '<code><![CDATA[', response)
    return response

def fix_cdata_issues(response):
    """
    修复 `CDATA` 内部的 `]]>`，防止解析错误
    """
    response = response.replace("]]>", "]] >")
    return response

def parse_xml(response: str) -> dict:
    """
    解析 XML 字符串并转换为字典，修正格式错误
    """
    # 预处理：去掉 GPT 生成的代码块标识符
    response = response.replace('```xml', '').replace('```', '').strip()

    # 执行 XML 修复
    response = fix_malformed_xml(response)
    response = clean_code_blocks(response)
    response = fix_cdata_issues(response)

    # 预处理：移除非法字符
    response = re.sub(r'[^\x09\x0A\x0D\x20-\x7F]', '', response)

    if not response:
        raise ValueError("Error: Empty XML response after cleanup")

    try:
        root = ET.fromstring(response)
    except ET.ParseError:
        print("First XML parsing failed. Attempting to wrap in <root>...")
        if "<root>" not in response:
            response = "<root>\n" + response.strip() + "\n</root>"
        try:
            root = ET.fromstring(response)
        except ET.ParseError as e:
            print("Final XML parsing failed.")
            print("Response causing error:", response)
            raise e

    return xml_to_dict(root)

def xml_to_dict(element):
    """
    递归地将 XML 解析为字典
    """
    result = {}
    for child in element:
        if child:
            child_data = xml_to_dict(child)
            if child.tag in result:
                if isinstance(result[child.tag], list):
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = [result[child.tag], child_data]
            else:
                result[child.tag] = child_data
        else:
            result[child.tag] = child.text.strip() if child.text else ""  # 确保去除空格
    return result


text = """
```xml
<root>
<problem>
<description><![CDATA[
Determine if a given number is a perfect square. A perfect square is an integer that is the square of an integer.
]]></description>
<code><![CDATA[
```python
def is_perfect_square(n):
    if n < 0:
        return False
    root = int(n**0.5)
    return root * root == n
```
]]></code>
<planning><![CDATA[
1. Check if the number is negative; if so, return False.
2. Calculate the integer square root of the number.
3. Square the integer root and check if it equals the original number.
4. Return True if it is a perfect square, otherwise return False.
]]></planning>
</problem>
<problem>
<description><![CDATA[
Check if a given string is a palindrome. A palindrome reads the same forwards and backwards.
]]></description>
<code><![CDATA[
```python
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]
```
]]></code>
<planning><![CDATA[
1. Convert the string to lowercase to ensure the check is case-insensitive.
2. Remove spaces from the string.
3. Compare the string to its reverse.
4. Return True if they are the same, otherwise return False.
]]></planning>
</problem>
<problem>
Count the number of vowels in a given string.
]]></description>
<code><![CDATA[
```python
def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count
```
]]></code>
<planning><![CDATA[
1. Define a string containing all vowels (both lowercase and uppercase).
2. Initialize a counter to zero.
3. Loop through each character in the input string.
4. If the character is a vowel, increment the counter.
5. Return the total count of vowels.
]]></planning>
</problem>
<algorithm><![CDATA[
The algorithm used to solve the original problem is a simple iterative check for primality, which can be considered a brute-force approach. This method involves checking divisibility of the number by all integers up to its square root.

A high-level tutorial for solving problems like this involves:
1. Understanding the problem requirements and constraints.
2. Identifying the properties of the numbers involved (e.g., prime numbers, perfect squares).
3. Developing a systematic approach to check conditions (like divisibility).
4. Implementing the solution in a programming language, ensuring to handle edge cases (like negative numbers or zero).
5. Testing the solution with various inputs to ensure correctness.
]]></algorithm>
</root>
"""


print(parse_xml(text))