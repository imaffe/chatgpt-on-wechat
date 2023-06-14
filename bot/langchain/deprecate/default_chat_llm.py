from chatgpt_tool_hub.models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.schema import HumanMessage


class DefaultConversationChain:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.0,
            model_name="gpt-3.5-turbo-0613")




    def reply_text(self, new_message, history):
        human_message = HumanMessage(content=new_message)
        # append human_message to history
        combined_history = history + [human_message]
        return self.llm(combined_history).content



