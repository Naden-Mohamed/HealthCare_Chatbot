from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",  # Free LLaMA 3 model
    temperature=1.0,  # Adjust temperature for creativity
    max_tokens=100,  # Limit response length
)

# Define messages
messages = [
    SystemMessage(content="YYou are a healthcare assistant. Provide medically accurate, patient-friendly answers based on WHO & Mayo Clinic guidelines. Always include a disclaimer."),
    HumanMessage(content="Hello! what is the anemia symptoms?")
]

# Get response
response = llm.invoke(messages)

print("Chatbot:", response.content)
