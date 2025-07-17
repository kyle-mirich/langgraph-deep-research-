from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from my_agent.utils.nodes import _get_model
from my_agent.utils.state import AgentState
from my_agent.utils.tools import search_tools, math_tools, tools


def supervisor_node(state: AgentState):
    """Supervisor node that analyzes the request and passes it along unchanged."""
    # Just pass the state through - the routing happens in conditional edges
    return state


def route_to_agent(state: AgentState) -> Literal["research_agent", "math_agent", "end"]:
    """Route to appropriate agent based on the message content."""
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Handle both message objects and dictionaries
    if hasattr(last_message, 'content'):
        content = last_message.content.lower()
    elif isinstance(last_message, dict) and 'content' in last_message:
        content = last_message['content'].lower()
    else:
        return "research_agent"  # Default fallback
    
    # Check for math-related keywords
    math_keywords = ["calculate", "math", "multiply", "divide", "add", "subtract", 
                     "plus", "minus", "times", "equation", "solve", "=", "+", "-", "*", "/"]
    
    # Check for research-related keywords  
    research_keywords = ["search", "find", "look up", "research", "who is", "what is", 
                        "when did", "where is", "how many", "current", "latest"]
    
    has_math = any(keyword in content for keyword in math_keywords)
    has_research = any(keyword in content for keyword in research_keywords)
    
    # If both or neither, default to research for now
    if has_math and not has_research:
        return "math_agent"
    else:
        return "research_agent"


def research_agent_node(state: AgentState):
    """Research agent node that handles web searches."""
    model = _get_model(state.get("model_name", "google"))
    
    # Add system message for research agent
    system_msg = {
        "role": "system", 
        "content": (
            "You are a research agent specialized in web search and information gathering. "
            "Use the search tools available to find current, accurate information. "
            "Provide detailed answers with sources when possible."
        )
    }
    
    # Bind only search tools to the model
    model_with_tools = model.bind_tools(search_tools)
    response = model_with_tools.invoke([system_msg] + state["messages"])
    
    return {"messages": [response]}


def math_agent_node(state: AgentState):
    """Math agent node that handles calculations."""
    model = _get_model(state.get("model_name", "google"))
    
    # Add system message for math agent
    system_msg = {
        "role": "system",
        "content": (
            "You are a math agent specialized in numerical calculations. "
            "Use the math tools available to perform accurate calculations. "
            "Show your work step by step and provide clear final answers."
        )
    }
    
    # Bind only math tools to the model
    model_with_tools = model.bind_tools(math_tools)
    response = model_with_tools.invoke([system_msg] + state["messages"])
    
    return {"messages": [response]}


def should_continue_agent(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue with tool calls or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Handle both message objects and dictionaries
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    elif isinstance(last_message, dict) and last_message.get('tool_calls'):
        return "tools"
    return "end"


# Create tool nodes for each agent type
research_tool_node = ToolNode(search_tools)
math_tool_node = ToolNode(math_tools)


def create_multi_agent_graph():
    """Create the multi-agent graph using LangGraph structure."""
    from typing import TypedDict, Literal
    
    # Define the config locally to avoid circular import
    class GraphConfig(TypedDict):
        model_name: Literal["anthropic", "openai","google"]
    
    # Define the multi-agent workflow
    workflow = StateGraph(AgentState, config_schema=GraphConfig)
    
    # Add supervisor node that routes to appropriate agent
    workflow.add_node("supervisor", supervisor_node)
    
    # Add research agent and its tools
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("research_tools", research_tool_node)
    
    # Add math agent and its tools  
    workflow.add_node("math_agent", math_agent_node)
    workflow.add_node("math_tools", math_tool_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor to agents
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research_agent": "research_agent",
            "math_agent": "math_agent",
            "end": END,
        },
    )
    
    # Add conditional edges from agents to tools or end
    workflow.add_conditional_edges(
        "research_agent",
        should_continue_agent,
        {
            "tools": "research_tools",
            "end": END,
        },
    )
    
    workflow.add_conditional_edges(
        "math_agent", 
        should_continue_agent,
        {
            "tools": "math_tools",
            "end": END,
        },
    )
    
    # Add edges from tools back to agents
    workflow.add_edge("research_tools", "research_agent")
    workflow.add_edge("math_tools", "math_agent")
    
    return workflow.compile()


# Create the compiled multi-agent graph
multi_agent_graph = create_multi_agent_graph()