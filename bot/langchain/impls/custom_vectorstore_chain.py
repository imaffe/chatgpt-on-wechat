import pathlib

from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from common.log import logger


class VectorStoreChain:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.vector_db_chain = self.create_vector_db_chain(openai_api_key)

    @classmethod
    def create_vector_db_chain(cls, openai_api_key):
        logger.info("[VectorStoreChain] Created VectorStoreChain")
        vector_db_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.0,
            model_name="gpt-3.5-turbo")

        logger.info("[VectorStoreChain] Creating docsearch")
        docsearch = cls.create_doc_search(openai_api_key=openai_api_key)

        # question_prompt
        question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
        Return any relevant text in Chinese.
        {context}
        Question: {question}
        Relevant text, if any, in Chinese:"""
        question_prompt = PromptTemplate(
            template=question_prompt_template, input_variables=["context", "question"]
        )

        combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer Chinese. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Answer in Chinese:"""
        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question"]
        )

        chain_type_kwargs = {"question_prompt": question_prompt, "combine_prompt": combine_prompt}
        db_chain = RetrievalQA.from_chain_type(
            llm=vector_db_llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            #chain_type_kwargs=chain_type_kwargs,
        )
        logger.info("[VectorStoreChain] Created VectorStoreChain")
        return db_chain
    @classmethod
    def create_doc_search(cls, openai_api_key=None):
        # the csv file path using pathlib and convert to string
        parent_path = pathlib.Path(__file__).parent.parent.absolute()
        logger.info("[VectorStoreChain] parent_path={}".format(parent_path.as_posix()))
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
        return docsearch

    def reply_text(self, user_question):
        answer = self.vector_db_chain.run(user_question)
        return answer