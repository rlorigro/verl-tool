system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test the code of your solution, include \"<|calling system for feedback|>\" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. Make sure you also write test cases for the code you write so you can get non-empty execution results for debugging. If you think the solution is complete and don't need to test, don't include \"<|calling system for feedback|>\" in the response and put your final answer in a markdown code block like this: ```python\nyour code here\n``` without appending anything.
"""

# math_problem = """## Find Events Within Date Range\n\nYou are given a list of events, each represented by a tuple containing the event's name and its date in ISO 8601 format (`YYYY-MM-DD`). Write a function `find_events_within_range(events, start_date, end_date)` that takes the following parameters:\n\n- `events`: A list of tuples, where each tuple consists of a string (event name) and a string (event date in `YYYY-MM-DD` format).\n- `start_date`: A string representing the start date in `YYYY-MM-DD` format.\n- `end_date`: A string representing the end date in `YYYY-MM-DD` format.\n\nThe function should return a list of event names that occur within the specified date range, **inclusive** of the start and end dates. If either `start_date` or `end_date` is not a valid date in the `YYYY-MM-DD` format, the function should return an empty list. You can assume that all dates in the `events` list are valid.\n\n### Example 1:\n\n**Input:**\n```python\nevents = [\n    (\"Event1\", \"2023-01-01\"),\n    (\"Event2\", \"2023-05-15\"),\n    (\"Event3\", \"2023-07-20\")\n]\nstart_date = \"2023-05-01\"\nend_date = \"2023-06-30\"\n```\n\n**Output:**\n```python\n[\"Event2\"]\n```\n\n### Example 2:\n\n**Input:**\n```python\nevents = [\n    (\"Conference\", \"2022-12-12\"),\n    (\"Meetup\", \"2023-03-22\"),\n    (\"Workshop\", \"2023-03-22\"),\n    (\"Webinar\", \"2023-04-10\")\n]\nstart_date = \"2023-03-22\"\nend_date = \"2023-03-22\"\n```\n\n**Output:**\n```python\n[\"Meetup\", \"Workshop\"]\n```
# """
math_problem = """Please write an print for hello world. But you are not sure about the python version of the env. You may first try python 2 version's print syntax"""

from openai import OpenAI
client = OpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000") # Replace with your local server address
model_name="Qwen/Qwen2.5-Coder-7B-Instruct"

completion = client.chat.completions.create(
    model=model_name,
    messages=[
		# {
        #     "role": "system",
        #     "content": system_prompt
        # },
        {
            "role": "user",
            "content": math_problem
        }
    ],
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].message.content)

tempalte = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{math_problem}
<|im_end|>
<|im_start|>assistant
"""
completion = client.completions.create(
    model=model_name,
    prompt=tempalte.format(system_prompt=system_prompt, math_problem=math_problem),
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)


completion = client.completions.create(
    model=model_name,
    prompt=f"system\n{system_prompt}\n\nuser\n{math_problem}\nassistant\n",
    temperature=0,
    max_tokens=1241241,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)