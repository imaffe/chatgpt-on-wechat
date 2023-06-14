from typing import Dict, Type, Any

import openai
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import OutputParserException, BaseOutputParser

from bot.langchain.impls.custom_search_chain import KeywordSearchSummaryChain
from bot.langchain.impls.custom_vectorstore_chain import VectorStoreChain
from bot.langchain.deprecate.default_chat_llm import DefaultConversationChain

CHAIN_SEARCH = "search"
CHAIN_VECTORDB = "vectordb"
CHAIN_DEFAULT = "DEFAULT"

routing_config_dict = {
    CHAIN_SEARCH: "",
    CHAIN_VECTORDB: "",
}

class OpenaiRoutingChain:
    def __init__(
        self,
        openai_api_key: str,
        search_api_key: str,
    ):
        self.openai_api_key = openai_api_key

        self.vector_db_chain = VectorStoreChain(openai_api_key=openai_api_key)
        self.search_chain = KeywordSearchSummaryChain(openai_api_key=openai_api_key, search_api_key=search_api_key)
        self.default_chain = DefaultConversationChain(openai_api_key=openai_api_key)
    def decide_routing(self, openai_api_key, new_message):
        response = openai.ChatCompletion.create(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": new_message}],
            functions=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call="auto",
        )


    def route(self, new_message, history):
        result = self.routing_chain.run({"input": new_message})
        destination = result["destination"]
        next_inputs = result["next_inputs"]

        if destination == CHAIN_VECTORDB:
            return self.vector_db_chain.reply_text(next_inputs)
        elif destination == CHAIN_SEARCH:
            return self.search_chain.reply_text(next_inputs)
        elif destination == CHAIN_DEFAULT:
            return self.default_chain.run(next_inputs, history)
        else:
            raise ValueError(f"Unknown destination: {destination}")

