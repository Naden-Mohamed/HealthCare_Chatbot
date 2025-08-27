from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)


def normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            normalized.append(HumanMessage(content=m["content"]))
        elif hasattr(m, "content"):
            normalized.append(HumanMessage(content=m.content))
    return normalized


def call_model(state: MessagesState, query: str, top_chunks: list, llm=None):
    if llm is None:
        raise ValueError("LLM instance must be provided")

    system_message = SystemMessage(content="You are a helpful medical assistant.")
    user_message = HumanMessage(content=query)
    context_message = HumanMessage(content="\n\n".join(top_chunks))
    message_history = normalize_messages(state["messages"])

    # Summarize old messages if too long
    if len(message_history) >= 4:
        summary_prompt = "Summarize the previous messages concisely."
        summary_output = llm(message_history + [HumanMessage(content=summary_prompt)])
        summary_message = HumanMessage(content=getattr(summary_output, "content", str(summary_output)))
        message_updates = [summary_message, context_message, user_message]
    else:
        message_updates = [system_message, context_message, user_message]


    response = llm(message_updates)
    response_message = HumanMessage(content=getattr(response, "content", str(response)))

    # Update state with new messages
    state["messages"].extend(message_updates + [response_message])
    return response_message.content




workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

