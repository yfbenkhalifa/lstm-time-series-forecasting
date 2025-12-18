from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(
    base_url="http://192.168.1.7:25000/v1",
    api_key="not-needed",
    model="google/gemma-3n-e4b",
    temperature=0.7
)

language_teacher_prompt = (
    "You are a patient, encouraging language teacher. Your job is to help the user learn a target language "
    "(TL) through conversation and guided practice. The idea is to respond to the user's query with the target language."
    "\n"
    "\n"
    "Rules:\n"
    "- Always converse primarily in the TL.\n"
    "- For every sentence in the foreign language you prompt, give the English translation for that if the user is at a beginner level.\n"
    "- For every sentence given by the user, analyze it and specify any correction or specification that need to be made.\n"
    "- Corrections and specifications MUST BE PROMPTED IN ENGLISH NOT IN GERMAN, so the user can understand its mistakes.\n"
    "- Highlight in chat with the keyword CORRECTION and specifications with the keyword SPECIFICATION.\n"
    "- When the user's response is not correct or not used appropriately based on the conversation, respond in english by correcting the user mistake.\n"
    "- Keep responses short and clear (2-5 sentences), suited to the user's level.\n"
    "- If the user uses another language, respond in the TL and optionally mirror key phrases in the user's language.\n"
    "- Correct mistakes gently: first show the corrected TL sentence, then briefly explain the error.\n"
    "- Highlight useful vocabulary, grammar points, and pronunciation tips when relevant.\n"
    "- Encourage active recall: ask simple questions and prompt the user to respond.\n"
    "- The user may use words in \"quotes\" like this if it does not know the correct translation of the work he wants to use.\n"
    "- Adapt difficulty: if the user struggles, simplify; if they perform well, gradually increase complexity.\n"
    "- Provide examples and mini-drills (fill-in-the-blank, rephrase, role-play) when helpful.\n"
    "- Avoid long lectures; prioritize interactive practice.\n"
    "- Ask the user for their TL and proficiency level at the start if not provided.\n"
    "\n"
    "Output format:\n"
    "- TL message\n"
    "- (Optional) brief correction/explanation in the user's language\n"
    "- Short follow-up question in TL to continue the conversation."
)

# Build conversation with context
initial_chat = [
    SystemMessage(content=language_teacher_prompt),
    AIMessage(content="What language would you like to learn, and what is your current proficiency level?")
]

# Prompt with a placeholder for past messages
prompt = ChatPromptTemplate.from_messages([
    *initial_chat,
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain,
    lambda session_id: get_history(session_id),
    input_messages_key="input",
    history_messages_key="history",
)

def build_messages(user_input: str, session_id: str) -> list:
    history = get_history(session_id)
    messages = [
        SystemMessage(content=language_teacher_prompt),
        *history.messages,
        HumanMessage(content=user_input)
    ]
    return messages


def run_chat(session_id: str = "default"):
    print("Type 'exit' to quit.")

    # Extract and display the initial AI message from initial_chat array
    initial_ai_message = next((msg.content for msg in initial_chat if isinstance(msg, AIMessage)), None)
    if initial_ai_message:
        print(f"Assistant: {initial_ai_message}")

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break

        messages = build_messages(user, session_id)
        resp = llm.invoke(messages)

        # Store the conversation
        history = get_history(session_id)
        history.add_message(HumanMessage(content=user))
        history.add_message(AIMessage(content=resp.content))

        print(f"Assistant: {resp.content}")


if __name__ == "__main__":
    run_chat("session-1")