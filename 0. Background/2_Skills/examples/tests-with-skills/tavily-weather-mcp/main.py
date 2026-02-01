from fastmcp import FastMCP
from tools.search_tool import TavilySearchTool

def main():
    mcp = FastMCP("tavily-weather-mcp")
    mcp.add_tool(TavilySearchTool())
    mcp.run()

if __name__ == "__main__":
    main()