- [Replace-OpenAI-model-to-llama-in-Ollama](#Replace-OpenAI-model-to-llama-in-Ollama)
- [1-Chat-model-basic](#1-Chat-model-basic)
- [2-chat-model-basic-conversation](#2-chat-model-basic-conversation)
- [3-Chat-model-alternatives](#3-Chat-model-alternatives)

## Replace-OpenAI-model-to-llama-in-Ollama

Since OpenAI cannot be used in China, maintaining a llama model in Ollama
```Python
from langchain_community.llms import Ollama
# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")
⬇️
model = Ollama(model="llama3.1:8b")
```


## 1-Chat-model-basic
Basic model invoke in langchain
```python
# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = Ollama(model="llama3.1:8b")

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)

```

## 2-chat-model-basic-conversation

import different conversation like SystemMessage/AIMessage/HumanMessage
```python
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = Ollama(model="llama3.1:8b")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result}")

# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result}")

```

## 3-Chat-model-alternatives

``` python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama #New Add
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]


# # ---- LangChain OpenAI Chat Model Example ----

# # Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

# # Invoke the model with messages
# result = model.invoke(messages)
# print(f"Answer from OpenAI: {result.content}")


# # ---- Anthropic Chat Model Example ----

# # Create a Anthropic model
# # Anthropic models: https://docs.anthropic.com/en/docs/models-overview
# model = ChatAnthropic(model="claude-3-opus-20240229")

# result = model.invoke(messages)
# print(f"Answer from Anthropic: {result.content}")


# # ---- Google Chat Model Example ----

# # https://console.cloud.google.com/gen-app-builder/engines
# # https://ai.google.dev/gemini-api/docs/models/gemini
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# result = model.invoke(messages)
# print(f"Answer from Google: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatOllama(model="llama3.1:8b")

result = model.invoke(messages)
print(f"Answer from Ollama: {result.content}")


```

Practice for the Chat models alternatives. Learned langchain-ollama imported by poetry
1. add this line to pyproject.toml and save the file.
   ``langchain-ollama = ">=0.1.0,<0.2.0"``
2. run command `poetry lock` to add the latest changes.
3. run command `poetry install --no-root` to install new pkg

Compare with langchain_community.llms.Ollama in previous practice, chatOllama is recommended than it:

| **对比维度**           | **langchain_community.llms.Ollama** | **langchain_ollama.ChatOllama**        |
| ------------------ | ----------------------------------- | -------------------------------------- |
| **所属包**            | `langchain-community`（社区维护）         | `langchain-ollama`（官方专门维护）             |
| **基础类型**           | 继承 `BaseLLM`（文本生成模型）                | 继承 `BaseChatModel`（对话模型）               |
| **输入格式**           | 主要接收字符串（`str`）                      | 接收结构化消息列表（`HumanMessage`/`AIMessage`等） |
| **输出格式**           | 返回字符串（`str`）                        | 返回 `AIMessage` 等对话对象（含元数据）             |
| **核心功能**           | 基础文本补全                              | 对话管理、角色区分、历史记录跟踪                       |
| **高级特性支持**         | 有限（如流式输出支持较弱）                       | 完善（流式输出、工具调用、模型参数细调等）                  |
| **与 LangChain 集成** | 兼容基础链（如 `LLMChain`）                 | 兼容对话链（如 `ConversationChain`）、代理等       |
| **推荐使用场景**         | 简单文本生成、临时快速调用                       | 对话系统开发、复杂交互场景（如聊天机器人）                  |
| **维护优先级**          | 社区贡献为主，更新较慢                         | 官方重点维护，更新及时（跟进 Ollama 新特性）             |
