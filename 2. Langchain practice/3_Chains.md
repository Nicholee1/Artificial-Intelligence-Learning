## Chain Basic
Chain can combine prompt template, model and stdout in one line.
```python
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")
model = ChatOllama(model="llama3.1:8b")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)

```
Here is the output without StrOutputParser():
```text
content='Here we go! Here are three lawyer jokes for you:\n\n1. Why did the lawyer\'s dog go to the vet? Because it was feeling ruff in its contract.\n2. What did the lawyer say when his client asked him to take a case on contingency? "Sorry, buddy. I\'m only interested in cases with a 100% chance of getting me a free lunch."\n3. Why do lawyers make great partners... for other lawyers? Because they\'re already used to arguing with themselves! (ba-dum-tss)' response_metadata={'model': 'llama3.1:8b', 'created_at': '2025-08-05T01:26:01.253293Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 7363290041, 'load_duration': 700386208, 'prompt_eval_count': 31, 'prompt_eval_duration': 1026831625, 'eval_count': 109, 'eval_duration': 5635232458} id='run-f9eab0cd-4e4e-4133-984f-c3e0d6a927f3-0' usage_metadata={'input_tokens': 31, 'output_tokens': 109, 'total_tokens': 140}
```
Here is the output with StrOutputParser():
```text
Here we go! Here are three lawyer jokes:

1. Why did the lawyer's dog go to the vet?

Because it was feeling ruff in court!

2. What did the lawyer say when his client asked him to reduce his fees?

"I'll give you a bill-iant discount, but only if you promise not to sue me for it!"

3. Why do lawyers make great comedians?

Because they're already trained to twist the truth and make a joke out of anything! (ba-dum-tss)
```

## Chains under the hood

RunnableLambda is a flexible plugin in LangChain, it helps us to involve function and lambda expression to LangChain runnable object.
```python
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4")
model = ChatOllama(model="llama3.1:8b")
# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)

```