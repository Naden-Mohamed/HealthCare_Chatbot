from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState, query: str, top_chunks: list, llm=None):
    """
    state: MessagesState for chat memory
    query: user query
    top_chunks: top retrieved RAG chunks
    llm: LLM instance (ChatGroq)
    """

    if llm is None:
        raise ValueError("You must pass an LLM instance to call_model")

    system_prompt = (
        "You are a helpful medical assistant. "
        "Answer all questions to the best of your ability using the context provided."
    )
    system_message = SystemMessage(content=system_prompt)

    # Include retrieved chunks as context
    context_message = HumanMessage(content="\n\n".join(top_chunks))

    # Include latest user query
    user_message = HumanMessage(content=query)

    # If chat history exists, summarize older messages
    message_history = state["messages"][:-1] if state["messages"] else []
    if len(message_history) >= 4:
        # Summarize older messages
        summary_prompt = "Summarize the previous messages into a concise summary."
        summary_message = llm(message_history + [HumanMessage(content=summary_prompt)])
        # Delete older messages (optional)
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        message_updates = [summary_message, user_message, context_message] + delete_messages
    else:
        message_updates = [system_message, context_message, user_message]

    # Call LLM with messages
    response = llm(message_updates)

    # Store messages in state
    state["messages"].extend(message_updates)
    state["messages"].append(HumanMessage(content=response.content if hasattr(response, "content") else response))

    return {"messages": state["messages"]}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)