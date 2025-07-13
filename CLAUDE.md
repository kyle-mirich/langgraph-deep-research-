# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph Cloud example agent that demonstrates building stateful, multi-actor applications with LLMs. The agent uses a conversational flow with tool calling capabilities and is designed to be deployed to LangGraph Cloud.

## Architecture

The codebase follows a modular structure:

- **`my_agent/agent.py`**: Main graph definition using StateGraph with agent/action nodes and conditional edges
- **`my_agent/utils/state.py`**: Defines AgentState TypedDict with message handling via add_messages
- **`my_agent/utils/nodes.py`**: Contains core nodes (call_model, should_continue) and model initialization with Google Gemini support
- **`my_agent/utils/tools.py`**: Tool definitions, currently using TavilySearchResults for web search

The agent follows a cycle: agent node → conditional routing → action node (if tools needed) → back to agent node until completion.

## Configuration

- **`langgraph.json`**: LangGraph Cloud deployment configuration pointing to `my_agent/agent.py:graph`
- **GraphConfig**: Supports model selection between "anthropic", "openai", and "google"
- **Environment**: Uses `.env` file for environment variables (API keys)

## Model Configuration

The current setup uses Google Gemini (`gemini-2.5-flash`) with these LLM providers supported:
- Google Generative AI (active)
- Anthropic Claude
- OpenAI

Model selection is handled through the GraphConfig and _get_model function with LRU caching.

## Dependencies

Key dependencies managed in `my_agent/requirements.txt`:
- `langgraph` - Core graph framework
- `langchain_anthropic` - Anthropic integration
- `langchain_openai` - OpenAI integration  
- `langchain_google_genai` - Google Gemini integration
- `tavily-python` - Web search tool
- `langchain_community` - Community tools

## Development Commands

This project uses requirements.txt for dependency management. To work with the codebase:

```bash
# Install dependencies
pip install -r my_agent/requirements.txt

# Run the agent locally (if applicable)
python my_agent/agent.py
```

## Deployment

The agent is designed for LangGraph Cloud deployment. The `langgraph.json` configuration defines the graph entry point and dependencies for cloud deployment.