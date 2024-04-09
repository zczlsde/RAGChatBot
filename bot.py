import os
import langchain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
import time
import logging
import requests
from typing import Optional, List, Dict, Mapping, Any

logging.basicConfig(level=logging.INFO)
langchain.llm_cache = InMemoryCache()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DialoGPT(LLM):
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    n = 5
    
    @property
    def _llm_type(self) -> str:
        return "DialoGPT"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        
        new_user_input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = self.model.generate(new_user_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        return {"n": self.n}


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    
    # Load Llama
    # llm = CTransformers(
    #     model = "TheBloke/Llama-2-7B-Chat-GGML",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5
    # )
    
    # Load ChatGPT
    os.environ["OPENAI_API_KEY"] = "" # Add your API key here
    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
    # gpt-4-turbo-preview
    # Load DialoGPT
    # llm = DialoGPT()
    
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa