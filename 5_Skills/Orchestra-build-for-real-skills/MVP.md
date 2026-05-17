通过简单的代码，让本地llm拥有调用工具执行工具的能力
```python
import requests
import json
from datetime import datetime


SYSTEM_PROMPT = """
您是一个智能助手，具备以下技能：
1. get_time: 获取当前时间
2. read_file(path): 读取文件

当需要使用技能时，请输出 JSON，
注意：只输出以下JSON结构，不要有其他的符号，包括markdown的```：
{
    "action": "skill_name",
    "args": {...}
}
否则直接回答用户问题。
"""
def call_llm(messages):
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3.5:9b",
            "messages": messages,
            "stream": False
        }
    )
    return res.json()["message"]["content"]

def execute_skill(action, args):
    if action == "get_time":
        return str(datetime.now())
    elif action == "read_file":
        with open(args["path"]) as f:
            return f.read()

def agent(user_input):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    reply = call_llm(messages)

    # 判断是不是 skill 调用
    try:
        data = json.loads(reply)
        if "action" in data:
            result = execute_skill(data["action"], data.get("args", {}))

            # 再让模型总结
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "tool", "content": result})

            final = call_llm(messages)
            return final
    except:
        pass

    return reply


def main():
    while True:
        user_input = input("请输入指令（或输入 'exit' 退出）：")
        if user_input.lower() == 'exit':
            break
        response = agent(user_input)
        print(f"回复：{response}")

if __name__ == "__main__":
    main()

```