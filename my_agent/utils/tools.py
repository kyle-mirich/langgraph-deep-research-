from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class GoogleSearchInput(BaseModel):
    query: str = Field(description="The search query to search for")


class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = "Search Google for information using Google AI's search capability"
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(self, query: str) -> str:
        """Execute the Google search using GenAITool"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Initialize the LLM with Google search capability
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.0,
            google_api_key="AIzaSyAB0GcW2FFoCKiD0z4Mmk6P-fm0lrNfiRI"
        )
        
        # Use the GenAITool for Google search
        resp = llm.invoke(
            query,
            tools=[GenAITool(google_search={})],
        )
        
        return resp.content


tools = [ GoogleSearchTool()]