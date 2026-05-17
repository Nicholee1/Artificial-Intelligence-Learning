import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client import PythonStdioTransport

# Run main.py with same Python as this script; cwd so main.py finds config/tools
_ROOT = Path(__file__).resolve().parent

async def main():
    transport = PythonStdioTransport(
        script_path=_ROOT / "testMain.py",
        cwd=str(_ROOT),
        python_cmd=sys.executable,
    )
    async with Client(transport=transport) as client:
        result = await client.call_tool("tavily_search", {"query": "北京天气"})
        print(result)

asyncio.run(main())