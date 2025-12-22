import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class Chatter:
    session_id = str(uuid.uuid4())
    initial_chat = []
    store = {}
    chat_prompt : ChatPromptTemplate
    history: ChatMessageHistory

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.chain = self.prompt | llm
        self.chat = RunnableWithMessageHistory(
            self.chain,
            lambda session: self.history,
            input_messages_key="input",
            history_messages_key="history"
        )
        self.store[self.session_id] = ChatMessageHistory()
        self.history = self.store[self.session_id]

    def init_chat(self, language_and_proficiency: str):
        self.history.add_message(HumanMessage(content=language_and_proficiency))
        response =  self.chat.invoke({
            "input": language_and_proficiency,
        },
        config={"configurable": {"session_id": self.session_id}})

        return response


    def build_messages(self, user_input):
        messages = [
            *self.history.messages,
            HumanMessage(content=user_input)
        ]
        return messages

    def invoke(self, user_input: str):
        response = self.chat.invoke(
            {"input": user_input},
            config={"configurable":
                        {"session_id": self.session_id}
                    }
        )
        return response

