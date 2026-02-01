- [Background](#Background)
- [Practice](#Practice)

## Background
20260121, 阅读了卡兹克三篇关于Skills的讲解
[一文带你看懂，火爆全网的Skills到底是个啥。](https://mp.weixin.qq.com/s/nRVVqPaGxWdNqNrUcurSXg)

其实更加明确了，Skills并不是只某种multiple agent的搭建方式。
也不是简单的动态加载prompt。

它和MCP tools的区别，可以 有 progressive disclosure的概念。
也就是渐进式披露，需要哪些能力，渐进式的加载，而不是将MCP tool的所有规范和上下文，全部一股脑的放在prompt里当作context每次都传给llm，耗时耗力。

而Skill自身，也不局限于prompt的渐进式披露，也可以包含可执行的脚本/工具的动态披露。
让llm知道有哪些技能包，然后有选择的动态调用。

Reference：
[anthropics skills](https://github.com/anthropics/skills)
[国内的技能应用平台](https://www.coze.cn/skills)

## Practice
通过 anthropic 提供的官方skills中的MCP builder, 构建一个web search的MCP Server
https://github.com/anthropics/skills/tree/main/skills/mcp-builder

![[Skiil_MCP_Builder.png]]

其中只有一个脚手架scaffold是自己填写的，主要就是配置本地的ollama模型-qwen3
去加载这个Skill.md，相当于告诉model你应该怎么使用这个mcp-builder
```python
"""
MCP Builder Skill scaffold — 使用本地 Ollama 运行 MCP Builder 技能。

解析 SKILL.md（YAML frontmatter + Markdown 流程说明），将用户请求与技能描述
一并交给 Ollama 模型，由模型按技能指南给出建议或步骤。
"""

import argparse
import os
import re
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("请先安装 ollama Python 客户端: pip install ollama")

# 默认使用的 Ollama 模型，可通过环境变量 OLLAMA_MODEL 或 --model 覆盖
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _skill_dir() -> Path:
    """返回 scaffold 所在目录（即 mcp-builder skill 根目录）。"""
    return Path(__file__).resolve().parent


def parse_skill_md(skill_md_path: str | Path | None = None) -> dict:
    """
    解析 SKILL.md：提取 YAML frontmatter（name, description, license）及正文内容。

    Args:
        skill_md_path: SKILL.md 路径，默认使用与 scaffold 同目录下的 SKILL.md。

    Returns:
        包含 name, description, license, content 的字典。
    """
    path = Path(skill_md_path) if skill_md_path else _skill_dir() / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"未找到技能文件: {path}")

    text = path.read_text(encoding="utf-8")

    # 解析 YAML frontmatter (--- ... ---)
    front_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    config = {}
    if front_match:
        front = front_match.group(1)
        for line in front.strip().split("\n"):
            m = re.match(r"^(\w+):\s*(.*)$", line.strip())
            if m:
                config[m.group(1).strip()] = m.group(2).strip()
        content = text[front_match.end() :].strip()
    else:
        content = text.strip()

    config["content"] = content
    return config


def run_skill_with_ollama(
    skill_config: dict,
    user_input: str,
    model: str | None = None,
) -> str:
    """
    使用本地 Ollama 模型按 MCP Builder 技能指南响应用户请求。

    Args:
        skill_config: parse_skill_md() 返回的配置（含 name, description, content）。
        user_input: 用户的问题或需求（例如：帮我生成一个 MCP Server，用于调用计算器工具）。
        model: Ollama 模型名，默认使用 DEFAULT_MODEL（或环境变量 OLLAMA_MODEL）。

    Returns:
        模型生成的回复文本。
    """
    model = model or DEFAULT_MODEL
    name = skill_config.get("name", "mcp-builder")
    description = skill_config.get("description", "")
    content = skill_config.get("content", "")

    # 将技能流程与用户输入一起交给模型
    prompt = f"""你正在使用「{name}」技能。该技能的说明如下：

## 技能描述
{description}

## 技能流程与参考（请按此指南回答）
{content[:12000]}

---

用户请求：
{user_input}

请严格按照上述 MCP Builder 技能指南，针对用户请求给出具体、可执行的建议或步骤（例如如何设计工具、项目结构、调用方式等）。若需生成代码，请直接写出可用代码片段。"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.get("message", {}).get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用本地 Ollama 运行 MCP Builder 技能（解析 SKILL.md 并调用模型）"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama 模型名（默认: {DEFAULT_MODEL}，或环境变量 OLLAMA_MODEL）",
    )
    parser.add_argument(
        "--input", "-i",
        default="帮我生成一个 MCP Server，用于调用计算器工具",
        help="用户请求内容",
    )
    parser.add_argument(
        "--skill-file",
        default=None,
        help="SKILL.md 路径（默认使用与 scaffold 同目录的 SKILL.md）",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="仅解析并打印技能配置，不调用 Ollama",
    )
    args = parser.parse_args()

    skill_path = args.skill_file or (_skill_dir() / "SKILL.md")
    skill_config = parse_skill_md(skill_path)

    print("解析到的技能配置：")
    for k in ("name", "description", "license"):
        if k in skill_config:
            print(f"  {k}: {skill_config[k]}")
    print()

    if args.no_run:
        print("（已使用 --no-run，跳过 Ollama 调用）")
        return

    print(f"使用模型: {args.model}")
    print(f"用户请求: {args.input}")
    print("-" * 40)
    result = run_skill_with_ollama(skill_config, args.input, model=args.model)
    print("技能执行结果：")
    print(result)


if __name__ == "__main__":
    main()

```

然后你就会得到
```
python scaffold.py --model qwen3:latest --input "帮我生成一个使用tavily-search的提供web search的MCP SERVER，只提供核心能力，不包含跨域和认证等功能，保证引入包和类的准确性，python实现"
```
=======================================================================
以下是输出
=======================================================================

解析到的技能配置：
  name: mcp-builder
  description: Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK).
  license: Complete terms in LICENSE.txt

使用模型: qwen3:latest
用户请求: 帮我生成一个使用tavily-search的提供web search的MCP SERVER，只提供核心能力，不包含跨域和认证等功能，保证引入包和类的准确性，python实现
----------------------------------------
技能执行结果：
以下是使用 Tavily Search 实现的 Python MCP 服务器核心代码，严格遵循 MCP Builder 技术指南：

```python
# main.py
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from mcp import McpError
from mcp.types import ErrorData

# 1. 定义工具输入输出 schema
class SearchQuery(BaseModel):
    query: str = Field(description="用户搜索的查询语句", example="AI模型的最新进展")

class SearchResult(BaseModel):
    results: list = Field(description="搜索结果列表，包含标题、链接和摘要")
    total_results: int = Field(description="总结果数量")

# 2. 实现搜索工具
def tavily_search(query: SearchQuery) -> SearchResult:
    """
    使用 Tavily Search API 进行网络搜索
    
    参数:
        query: 用户搜索的查询语句
        
    返回:
        搜索结果列表，包含标题、链接和摘要
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise McpError(ErrorData(code=-32602, message="TAVILY_API_KEY 环境变量未设置", data=None))
    
    BASE_URL = "https://api.tavily.com/search"
    try:
        q = query.query if isinstance(query, SearchQuery) else query
        response = requests.post(
                BASE_URL,
                json={"query": q},
                headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return SearchResult(results=data.get("results", []), total_results=len(data.get("results", [])))
    except requests.RequestException as e:
        raise McpError(ErrorData(code=-32603, message=f"网络请求失败: {str(e)}", data=None))

# 3. 初始化 MCP 服务器
app = FastAPI()

# 4. 注册工具到 MCP 服务器
@app.get("/tools")
def list_tools():
    """列出所有可用工具"""
    return {
        "tools": [
            {
                "name": "tavily_search",
                "description": "使用 Tavily Search API 进行网络搜索",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                "returns": {"type": "object", "properties": {"results": [{"type": "object", "properties": {"title": {"type": "string"}, "url": {"type": "string"}, "content": {"type": "string"}}}], "total_results": {"type": "integer"}}}
            }
        ]
    }

# 5. 处理 MCP 请求
@app.post("/mcp")
def handle_mcp_request(data: dict):
    """处理 MCP 请求并返回结果"""
    try:
        # 调用工具
        result = tavily_search(SearchQuery(**data))
        return {
            "content": f"找到 {result.total_results} 个相关结果:\n" + 
                      "\n".join([f"{i+1}. [{r['title']}]({r['url']}) - {r.get('content', '')[:100]}..." for i, r in enumerate(result.results[:5])]) 
        }
    except McpError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}

# 6. 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
```

**使用说明：**

1. 安装依赖：
```bash
pip install fastapi uvicorn pydantic requests
```

2. 设置环境变量：
```bash
export TAVILY_API_KEY="your_tavily_api_key"
```

3. 运行服务器：
```bash
python main.py
```

4. 使用 MCP 客户端调用：
```bash
"""
使用 MCP 服务器进行 Tavily 搜索的客户端脚本。
用法:
  python query_mcp.py "搜索关键词"
  python query_mcp.py   # 无参数时进入交互模式
"""
import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client import PythonStdioTransport

_ROOT = Path(__file__).resolve().parent


def format_result(result) -> str:
    """将 call_tool 返回结果格式化为可读文本"""
    lines = []
    if hasattr(result, "structured_content") and result.structured_content:
        data = result.structured_content
        results = data.get("results", [])
        if not results:
            return "未找到结果。"
        for i, r in enumerate(results, 1):
            if isinstance(r, dict):
                if "error" in r:
                    lines.append(f"  {i}. [错误] {r['error']}")
                else:
                    title = r.get("title", "(无标题)")
                    url = r.get("url", "")
                    content = (r.get("content") or "")[:150]
                    lines.append(f"  {i}. {title}")
                    lines.append(f"      {url}")
                    if content:
                        lines.append(f"      {content}...")
            else:
                lines.append(f"  {i}. {r}")
    elif hasattr(result, "content") and result.content:
        for block in result.content:
            if hasattr(block, "text"):
                lines.append(block.text)
    return "\n".join(lines) if lines else str(result)


async def run_query(client: Client, query: str) -> None:
    result = await client.call_tool("tavily_search", {"query": query})
    print(f"\n【查询】{query}")
    print("-" * 50)
    print(format_result(result))
    print()


async def main():
    transport = PythonStdioTransport(
        script_path=_ROOT / "main.py",
        cwd=str(_ROOT),
        python_cmd=sys.executable,
    )
    async with Client(transport=transport) as client:
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            await run_query(client, query)
        else:
            print("MCP 搜索客户端（输入查询后回车，空行退出）")
            print("-" * 50)
            while True:
                try:
                    query = input("查询> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not query:
                    break
                await run_query(client, query)


if __name__ == "__main__":
    asyncio.run(main())

```

**代码特点：**

1. 严格遵循 MCP 协议：
   - 使用 `@tool` 装饰器定义工具
   - 使用 Pydantic 定义输入输出 schema
   - 返回结构化数据（JSON 格式）

2. 核心功能实现：
   - 网络搜索功能
   - 错误处理机制
   - 工具发现接口（/tools）
   - MCP 请求处理接口（/mcp）

3. 安全性：
   - 使用环境变量存储敏感信息
   - 包含全面的异常处理

4. 扩展性：
   - 可通过添加更多工具扩展功能
   - 支持添加分页、过滤等高级功能

**注意事项：**
1. 需要自行替换 TAVILY_API_KEY 为实际的 API 密钥
2. 可通过修改 `result.results[:5]` 控制返回结果数量
3. 可添加更多工具来扩展 MCP 服务器功能
4. 生产环境需要添加更完善的日志和监控系统

=================================================
然后你就会获得一个websearch的MCP server
![[Web_search_Mcp.png]]

并可以通过client端进行访问
![[Web_search_Mcp_client.png]]