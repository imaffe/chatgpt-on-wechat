from typing import Dict, Type, Any

import openai
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import OutputParserException, BaseOutputParser

from bot.langchain.impls.custom_search_chain import KeywordSearchSummaryChain
from bot.langchain.impls.custom_vectorstore_chain import VectorStoreChain
from bot.langchain.impls.default_chat_llm import DefaultConversationChain

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the function best suited for \
the input. You will be given the names of the available function and a description of \
what the function is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the function to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate function names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate functions.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE FUNCTIONS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""


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



class RouterOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parser for output of router chain int he multi-prompt chain."""

    default_destination: str = "DEFAULT"
    next_inputs_type: Type = str
    next_inputs_inner_key: str = "input"

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            expected_keys = ["destination", "next_inputs"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if not isinstance(parsed["destination"], str):
                raise ValueError("Expected 'destination' to be a string.")
            if not isinstance(parsed["next_inputs"], self.next_inputs_type):
                raise ValueError(
                    f"Expected 'next_inputs' to be {self.next_inputs_type}."
                )
            parsed["next_inputs"] = {self.next_inputs_inner_key: parsed["next_inputs"]}
            if (
                parsed["destination"].strip().lower()
                == self.default_destination.lower()
            ):
                parsed["destination"] = None
            else:
                parsed["destination"] = parsed["destination"].strip()
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )
