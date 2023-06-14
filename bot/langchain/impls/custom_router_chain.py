from typing import Dict, Type, Any

from langchain import PromptTemplate
from langchain.chains.router import LLMRouterChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import OutputParserException, BaseOutputParser

from bot.langchain.impls.custom_search_chain import KeywordSearchSummaryChain
from bot.langchain.impls.custom_vectorstore_chain import VectorStoreChain
from bot.langchain.deprecate.default_chat_llm import DefaultConversationChain
from common.log import logger

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the prompt best suited for \
the input. You will be given the names of the available prompt and a description of \
what the prompt is best suited for. You may also revise the original input if you \
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

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""


CHAIN_SEARCH = "search"
CHAIN_VECTORDB = "vectordb"
CHAIN_DEFAULT = "DEFAULT"

routing_config_dict = {
    CHAIN_SEARCH: "Useful when you need to answer questions that you are not sure about.",
    CHAIN_VECTORDB: "Useful when you need to answer questions about a Outdoors sports company 飓风行",
}

class CustomRoutingChain:
    def __init__(
        self,
        openai_api_key: str,
        search_api_key: str,
    ):
        self.openai_api_key = openai_api_key
        self.routing_chain = self.create_routing_chain(openai_api_key)

        self.vector_db_chain = VectorStoreChain(openai_api_key=openai_api_key)
        self.search_chain = KeywordSearchSummaryChain(openai_api_key=openai_api_key, search_api_key=search_api_key)
        self.default_chain = DefaultConversationChain(openai_api_key=openai_api_key)
    @classmethod

    def create_routing_chain(cls, openai_api_key):


        # format the destinations, in the format:<NAME>: <description>
        formatted_destinations = "\n".join([f"{name}: {description}" for name, description in routing_config_dict.items()])
        # add the destinations

        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=formatted_destinations
        )

        routing_prompt_with_destinations = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser()
        )

        routing_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.0,
            model_name="gpt-3.5-turbo")
        routing_chain = LLMRouterChain.from_llm(llm=routing_llm, prompt=routing_prompt_with_destinations)
        return routing_chain

    def reply_text(self, new_message, history):
        result = self.routing_chain({"input": new_message})
        destination = result["destination"]
        raw_inputs = result["next_inputs"]

        logger.info("The destination is: {}, input: {}".format(destination, raw_inputs))
        next_inputs = raw_inputs['input']
        if destination == CHAIN_VECTORDB:
            return self.vector_db_chain.reply_text(next_inputs)
        else:
            return self.search_chain.reply_text(next_inputs)




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
