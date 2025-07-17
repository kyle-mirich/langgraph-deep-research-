#!/usr/bin/env python3
"""Test script for the multi-agent system."""

from my_agent.agent import graph


def test_math_query():
    """Test the math agent with a calculation."""
    print("=== Testing Math Query ===")
    query = "Calculate 15 + 25 and then multiply by 3"
    
    try:
        from langchain_core.messages import HumanMessage
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Final result: {result['messages'][-1].content}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_research_query():
    """Test the research agent with a web search."""
    print("=== Testing Research Query ===")
    query = "Who is the current mayor of New York City?"
    
    try:
        from langchain_core.messages import HumanMessage
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Final result: {result['messages'][-1].content}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def test_routing():
    """Test the routing functionality."""
    print("=== Testing Routing ===")
    
    # Test math routing
    from my_agent.multi_agent import route_to_agent
    from langchain_core.messages import HumanMessage
    
    math_state = {"messages": [HumanMessage(content="Calculate 5 + 3")]}
    math_route = route_to_agent(math_state)
    print(f"Math query routed to: {math_route}")
    
    # Test research routing  
    research_state = {"messages": [HumanMessage(content="Who is the president?")]}
    research_route = route_to_agent(research_state)
    print(f"Research query routed to: {research_route}")
    print()


if __name__ == "__main__":
    print("Testing Multi-Agent System\n")
    
    try:
        test_routing()
        test_math_query()
        test_research_query()
        print("All tests completed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()