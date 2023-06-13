import pathlib

from langchain import SerpAPIWrapper, LLMChain, OpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chains import RetrievalQA
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.utilities import BingSearchAPIWrapper
from langchain.vectorstores import Chroma



def process_new_message(chain=None, messages=None):
    new_message, _ = extract_memory(messages)
    raw_response = chain.run(new_message)
    response = raw_response
    return response
def create_custom_mrkl_agent(openai_api_key, serp_api_key):
    # create all the tools, we have [search]
    # search = _create_search_api_tool(serp_api_key=serp_api_key)
    search = _create_bing_api_tool(bing_api_key=serp_api_key)
    # create the toolkit
    tools = [search]

    # create the prompt
    prompt = _create_agent_prompt(tools=tools)

    # create the llm, be careful it's a temperature 0 llm
    gpt_llm = OpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name="gpt-3.5-turbo")

    llm_chain = LLMChain(llm=gpt_llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor


def _create_agent_prompt(tools):
    prefix = """
    You are 飓风小野人， a male Chinese outdoors sports coach that knows a lot about the outdoors sports.
    The input is a goal or a question. 
    You should use tools to find information that are relevant to the input and generate answers to help the user achieve their goal.
    but always use Chinese, English is not preferred in the final output. You have access to the following tools:
    """

    suffix = """
    Begin! Remember to use Chinese and speak as a Chinese male! Do not sound like a robot.
    
    Question: {input}
    {agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    return prompt
def _create_search_api_tool(serp_api_key):
    params = {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "cn",
        "hl": "zh-cn",
    }
    search = SerpAPIWrapper(serpapi_api_key=serp_api_key, params=params)
    search_tool = Tool(
                name="Search",
                func=lambda x: search.run(x),
                # description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term. Output should include all the original links."
                description="当你需要回答当前世界发生的事件或者有关当前世界的事实性问题时使用。输入的数据应该是一个搜索关键词。当时使用该工具时，你的输出应该包含搜索网站的源地址。",
            )
    return search_tool


def _create_bing_api_tool(bing_api_key):
    params = {
        "q": "query",
        "count": 10,
        "offset": 0,
        "mkt": "zh-CN",
        "safeSearch": "Moderate",
        "textDecorations": True,
        "textFormat": "HTML",
    }
    search = BingSearchAPIWrapper(
        bing_subscription_key=bing_api_key,
        bing_search_url="https://api.bing.microsoft.com/v7.0/search")
    search_tool = Tool(
                name="Search",
                func=lambda x: search.results(x, 1),
                description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term. Output is a json string, you should"
                # description="当你需要回答当前世界发生的事件或者有关当前世界的事实性问题时使用。输入的数据应该是一个搜索关键词。当时使用该工具时，你的输出应该包含搜索网站的源地址。",
            )
    return search_tool



def _create_doc_search_tool(openai_api_key=None):
    doc_llm = OpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    # the csv file path using pathlib and convert to string
    parent_path = pathlib.Path(__file__).parent.absolute()
    file_path = pathlib.Path(parent_path, "jufengxing.docx").absolute().as_posix()
    persist_path = pathlib.Path(parent_path, "chroma").absolute().as_posix()
    # create loader
    loader = Docx2txtLoader(file_path=file_path)
    documents = loader.load()
    # create text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # create embeddings and persist
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docsearch = Chroma.from_documents(documents=texts, embedding=embeddings,
                                      collection_name="jufengxing",
                                      persist_directory=persist_path)

    db_chain = RetrievalQA.from_chain_type(llm=doc_llm, chain_type="stuff", retriever=docsearch.as_retriever())
    return db_chain
