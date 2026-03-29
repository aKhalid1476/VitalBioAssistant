"""
RAG chain: conversational retrieval over the bloodborne pathogen regulation.

Exports a single module-level `rag_chain` singleton built at import time.
"""

from dotenv import load_dotenv

load_dotenv()

from config import settings
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

SYSTEM_PROMPT = (
    "You are VitalBio Assistant, a compliance expert specializing exclusively in "
    "OSHA Regulation 29 CFR § 1910.1030 – Bloodborne Pathogens.\n\n"
    "Your role is to help employers, healthcare workers, and safety officers understand "
    "their obligations and rights under this regulation.\n\n"
    "Guidelines:\n"
    "- Answer ONLY from the provided regulatory context below. Do not use outside knowledge.\n"
    "- Always cite the specific section (e.g., 'per § 1910.1030(c)(1)(ii)') when referencing requirements.\n"
    "- If the context does not contain enough information to answer, state clearly: "
    "'I cannot find that specific requirement in the regulation. Please consult the full OSHA "
    "standard or a qualified safety professional.'\n"
    "- For procedural requirements (PPE steps, post-exposure procedures, vaccination), use numbered lists.\n"
    "- Do not provide medical advice or diagnoses.\n"
    "- Be precise and use regulatory language where appropriate.\n\n"
    "Context from the regulation:\n{context}"
)

_CONDENSE_SYSTEM = (
    "Given the conversation history and a follow-up question, rephrase the follow-up "
    "into a self-contained standalone question that captures all necessary context. "
    "Output only the rephrased question, nothing else."
)


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key,
    )


def _build_retriever():
    embedding = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    chroma = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding,
        persist_directory=settings.chroma_persist_dir,
    )
    return chroma.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": 20,
            "lambda_mult": 0.6,
        },
    )


def build_conversational_rag_chain() -> RunnableWithMessageHistory:
    llm = _build_llm()
    retriever = _build_retriever()

    # History-aware retriever: rewrites follow-up questions before vector search
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", _CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=condense_prompt,
    )

    # QA chain: stuffs retrieved documents into the system prompt context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    # Full retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )

    # In-memory session store (swap for RedisChatMessageHistory for multi-instance deployments)
    session_store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    return RunnableWithMessageHistory(
        runnable=retrieval_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# Module-level singleton — initialized once at import time
rag_chain = build_conversational_rag_chain()
