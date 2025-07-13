from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class GoogleSearchInput(BaseModel):
    query: str = Field(description="The search query to search for")


class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = (
        "Search the internet and provide only raw factual information. "
        "Return relevant excerpts (paragraphs or snippets) from trustworthy sources. "
        "Always include the source URL, source name, and full quoted text. "
        "Do NOT summarize, paraphrase, or interpret — just extract useful chunks. "
        "Your job is to collect reliable web data for another AI agent to use."
    )
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(self, query: str) -> str:
        """Execute the Google search using GenAITool"""
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Embedded system prompt
        prompt = (
            "You are a web research tool. Search the internet and return only relevant text snippets "
            "from reputable sources. Do not summarize or paraphrase. Instead, extract raw text from the articles "
            "(such as full paragraphs or clearly relevant excerpts) and include the following for each result:\n\n"
            "- 🔗 Source URL\n"
            "- 📰 Source title or publisher\n"
            "- 📄 Full quoted snippet or excerpt\n\n"
            "Only include text that directly relates to the search query. Do not make anything up. "
            "If no relevant information is found, say so clearly."
        )

        # Initialize the LLM with Google search capability
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key="AIzaSyAB0GcW2FFoCKiD0z4Mmk6P-fm0lrNfiRI"
        )

        # Combine system prompt with user query
        full_query = f"{prompt}\n\nSearch query: {query}"

        # Perform the search
        resp = llm.invoke(
            full_query,
            tools=[GenAITool(google_search={})],
        )

        return resp.content


tools = [GoogleSearchTool()]
