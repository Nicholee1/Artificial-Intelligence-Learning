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
