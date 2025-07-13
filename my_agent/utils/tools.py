from langchain_community.tools.tavily_search import TavilySearchResults
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool


tools = [TavilySearchResults(max_results=1)]