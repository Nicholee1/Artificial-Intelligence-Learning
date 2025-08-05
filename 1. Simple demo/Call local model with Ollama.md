## Download Ollama and pull an AI model
```shell
licenhao@Nic-Lee AITest % ollama list

NAME           ID              SIZE      MODIFIED    

llama3.1:8b    46e0c10c039e    4.9 GB    6 hours ago
```

When the first api call, Ollama will call ``ollama serve`` implicitly.
``ollama serve`` run a model in port 11434
``ollama run`` run an interactive window with AI model
## In http request type

``` python
import requests
import os
  
try:
	response = requests.post(
	"http://localhost:11434/api/generate",
	json={"model": "deepseek-r1:7b", "prompt": "hello"},
	timeout=10
)
	print(response.status_code, response.text) # 应返回 200 和结果
except Exception as e:
	print(f"请求失败: {e}")
```

## In OpenAI type

```Python
from openai import OpenAI
os.environ["NO_PROXY"] = "localhost"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=30.0  # 关键：设置足够长的超时
)

try:
    # add your completion code
    prompt = "Pls provide a quick sort method in python"
    messages = [{"role": "user", "content": prompt}]
    # make completion

    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=messages
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"错误: {e}")
    print("请检查：1. Ollama 服务是否运行 2. /v1 端点是否启用 3. 模型是否存在")
```



## In Langchain type
1. ``pip install langchain langchain-community``
2. 
```Python
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化 Ollama 模型 (替换为你本地的模型名)
llm = Ollama(model="llama3.1:8b")  # 注意模型名格式可能与API调用时不同

# 定义提示模板 (LangChain推荐结构化提示)
prompt_template = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

# 创建处理链
chain = prompt_template | llm | StrOutputParser()

try:
    # 用户输入
    prompt = "Hello, who are you?"
    
    # 调用链并获取结果
    response = chain.invoke({"input": prompt})
    print(response)
    
except Exception as e:
    print(f"Error: {e}")
    print("请检查：1. Ollama 服务是否运行 2. 模型是否存在")   
```