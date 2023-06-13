from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.utilities import BingSearchAPIWrapper

class KeywordSearchSummaryChain:
    def __init__(self, openai_api_key, search_api_key):
        self.openai_api_key = openai_api_key
        self.search_api_key = search_api_key
        # TODO is
        self.keyword_chain = self.create_keyword_chain(openai_api_key)
        self.analyze_chain = self.create_analyze_chain(openai_api_key)
        self.search_wrapper = self.create_search_wrapper(search_api_key)

    @classmethod
    def create_keyword_chain(cls, openai_api_key):
        keyword_llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,
            model_name="gpt-3.5-turbo")

        keyword_prompt = """
        A Keyword is an input to an search engine such as Google or Bing. It is used to find relevant information for a topic.
        The following is a user question, you should summarize the question and figure out the most appropriate Keyword 
        related to the user question: 
        {user_question}
        
        THE KEYWORD HAS TO BE IN CHINESE.
        Please find out the most appropriate Keyword in Chinese:
        """

        prompt_template = PromptTemplate(template=keyword_prompt, input_variables=["user_question"])

        keyword_chain = LLMChain(llm=keyword_llm, prompt=prompt_template)
        return keyword_chain

    @classmethod
    def create_analyze_chain(cls, openai_api_key):
        analyze_llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,
            model_name="gpt-3.5-turbo")

        analyze_prompt = """
        User has asked the following question: 
        {user_question}
        
        
        You found some web pages from the search engine that might help you answer the user question. Here are the summaries from the pages you found.
    
        {summaries}
        
        You should analyze the summaries one by one and combine them with your own knowledge to answer the user question.
        Finally, you should politely ask the user if the answer is helpful. All output should be in Chinese.
        
        Your answer:
        """

        prompt_template = PromptTemplate(
            template=analyze_prompt,
            input_variables=["user_question", "summaries"])

        analyze_chain = LLMChain(llm=analyze_llm, prompt=prompt_template)
        return analyze_chain

    @classmethod
    def create_search_wrapper(cls, search_api_key):
        search = BingSearchAPIWrapper(
            bing_subscription_key=search_api_key,
            bing_search_url="https://api.bing.microsoft.com/v7.0/search")
        return search

    def retreive_search_results(self, user_question):
        # TODO this requires some error processing unit
        # TODO what is the output of the keyword chain ?
        keyword = self.keyword_chain.run({"user_question": user_question})
        # use tools
        search_results = self.search_wrapper.results(keyword, 3)
        return search_results

    def reply_text(self, user_question):
        search_results = self.retreive_search_results(user_question)
        # get the search result
        summaries = [result["snippet"] for result in search_results]
        titles = [result["title"] for result in search_results]
        links = [result["link"] for result in search_results]

        formatted_summary = "\n".join([f"Summary-{title}:{summary}" for title, summary in zip(titles, summaries)])

        analyze_result = self.analyze_chain.run({"user_question": user_question, "summaries": formatted_summary})

        return self.post_process_analyze_result(analyze_result, titles, links)


    def post_process_analyze_result(self, analyze_result, titles, links):
        # combine the titles and links in markdown link format, each line with an index, and append to analyze_result
        markdown = "\n".join([f"{index}. [{title}]({link})" for index, (title, link) in enumerate(zip(titles, links))])
        return analyze_result + "\n" + markdown
