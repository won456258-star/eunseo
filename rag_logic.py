# 파일명: rag_logic.py (API 키 사용 버전)
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

@st.cache_resource
def get_rag_chain():
    # 1. 벡터 저장소(Vector Store) 로드
    # DB를 로드할 때도 OpenAIEmbeddings를 사용합니다.
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "my_faiss_db", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()

    # 2. 프롬프트 템플릿 정의
    template = """
    당신은 '모구' 서비스에 대한 질문에 답변하는 친절한 AI 어시스턴트입니다.
    제공된 컨텍스트 정보만을 사용하여 사용자의 질문에 답변해 주세요.
    
    컨텍스트:
    {context}
    
    질문:
    {question}
    
    답변:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. 언어 모델(LLM) 초기화
    # Ollama 대신 다시 ChatOpenAI를 사용합니다.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 4. RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain