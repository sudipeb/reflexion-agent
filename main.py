from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessageState, StateGraph

from tool_executor import execute_tools
from chains import revisor, first_responder

MAX_ITERATIONS = 2


def draft_node(state: MessageState):
    """Draft the initial response with tool calls."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessageState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages":state["messages"]})
    return {"messages": [response]}


def _tool_message_count(state: MessageState) -> int:
    return sum(1 for msg in state["messages"] if isinstance(msg, ToolMessage))


def event_loop(state: MessageState) -> Literal["execute_tools", "__end__"]:
    """Continue looping while the model still asks for tools and we have iterations left."""
    latest = state["messages"][-1]
    if not isinstance(latest, AIMessage):
        return END

    has_tool_calls = bool(latest.tool_calls)
    iterations = _tool_message_count(state)

    if has_tool_calls and iterations < MAX_ITERATIONS:
        return "execute_tools"
    return END


builder = StateGraph(MessageState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop)

graph = builder.compile()


def run(question: str):
    inputs = {"messages": [HumanMessage(content=question)]}
    final_state = None

    for state in graph.stream(inputs, stream_mode="values"):
        final_state = state

    return final_state


if __name__ == "__main__":
    question = (
        "Write about AI-powered SOC/autonomous SOC problem domain, "
        "list startups working in this area and mention known funding."
    )
    result = run(question)
    if result and result.get("messages"):
        print(result["messages"][-1].content)