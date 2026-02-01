from fastmcp.tools.tool import Tool, ToolResult
from pydantic import BaseModel
import requests

from config.api_keys import TAVILY_API_KEY

BASE_URL = "https://api.tavily.com/search"


class SearchQuery(BaseModel):
    query: str


class SearchResults(BaseModel):
    results: list


class TavilySearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="tavily_search",
            description="Perform web search using Tavily API",
            parameters=SearchQuery.model_json_schema(),
            output_schema=SearchResults.model_json_schema(),
        )

    async def run(self, arguments: dict) -> ToolResult:
        try:
            query = arguments["query"]
            response = requests.post(
                BASE_URL,
                json={"query": query},
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            )
            response.raise_for_status()
            data = response.json()
            results = SearchResults(results=data.get("results", []))
            return ToolResult(structured_content=results.model_dump())
        except requests.exceptions.RequestException as e:
            return ToolResult(
                structured_content=SearchResults(results=[{"error": str(e)}]).model_dump()
            )
