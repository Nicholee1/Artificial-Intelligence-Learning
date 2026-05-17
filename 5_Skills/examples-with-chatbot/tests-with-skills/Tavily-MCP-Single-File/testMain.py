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