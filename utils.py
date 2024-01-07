from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain


def load_youtube_video(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    data = loader.load()
    return data

def split_text(data,chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

def initialise_llm(openai_api_key,model_name,temperature):
    llm = ChatOpenAI(openai_api_key=openai_api_key,model_name=model_name,temperature=temperature)
    return llm

def initialize_summarize_chain(llm,chain_type,question_prompt,refine_prompt):
    summarize_chain = load_summarize_chain(llm=llm,chain_type=chain_type,verbose=True,question_prompt=question_prompt,refine_prompt=refine_prompt)
    return summarize_chain

def get_summary(summarize_chain,docs):
    summary = summarize_chain.run(docs)
    return summary

def initialise_plan_chain(llm,prompt,verbose):
    plan_chain = LLMChain(llm=llm,prompt=prompt,verbose=verbose)
    return plan_chain

def generate_plan(plan_chain,summary):
    plan = plan_chain(summary)
    return plan