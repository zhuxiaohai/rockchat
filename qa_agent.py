from typing import Annotated, Literal
from typing_extensions import TypedDict
import functools
import json

from search_engine import QASearchEngine, VectorSim
from find_keywords import LabellerByRules
from recall import RecallBySearchEngine
from merge import QAMerge
from rank import QAScorer
from rerank import QAReranker
from pipeline import QAPineline

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import AzureChatOpenAI

from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


keywords_config = {
    "class": LabellerByRules,
    "config": {
        "dim_df_path": "data/dim_df20240315.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_name", "cat"),
        "error_col": ("error", "error"),
    }
}

recall_config = {
    "vector_search": {
        "class": RecallBySearchEngine,
        "config": {
            "search_engine": {
                "class": QASearchEngine,
                "database_path": "data/database20240506.csv",
                "id_col": "qa_id",
                "index_columns": [("model_list", "model"), ("cat_name", "cat"), ("error_list", "error")],
                "score_model": {
                    "type": "vector",
                    "class": VectorSim,
                    "embedding_col": "question",
                    "embedding_model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_emb"
                },
            },
        },
        "top_n": 10,
    }
}

merge_config = {
    "class": QAMerge,
    "config": {
        "vector_search": 1,
    }
}

rank_config = {
    "class": QAScorer,
    "config": {
        "model_path": "/workspace/data/private/zhuxiaohai/models/bge_finetune_reranker_question_top20",
        "query_key": "query_cleaned",
        "item_key": "question",
        "database_path": "data/database20240506.csv",
    }
}

rerank_config = {
    "class": QAReranker,
    "config": {
        "rank_key": [("rank", False)],
        "show_cols": ["question", "answer"],
        "reranking_scheme": {
            "recall_ranking_score_threshold": 0.75,
            "recall_ranking_top_n": 2,
        },
        "database_path": "data/database20240506.csv",
    }
}

pipeline_config = {
    "router": keywords_config,
    "recall": recall_config,
    "merger": merge_config,
    "ranker": rank_config,
    "reranker": rerank_config,
}

qa_pipeline = QAPineline(pipeline_config)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm_grader = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)

structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

# Prompt
system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


@tool
def search_qa_records(query):
    """Consult historical records of problems about household cleaning machines
 and return similar cases with the query"""
    results = qa_pipeline.run(query)[:3]
    pairs = [{"question": query, "document": d['question']} for d in results]
    scores = retrieval_grader.batch(pairs)
    filtered_docs = [results[i] for i in range(len(scores)) if scores[i].binary_score == "yes"]

    if not filtered_docs:
        return "搜索结果和问题的匹配度很低，请换个问法"
    else:
        return json.dumps({"question": filtered_docs[0]["question"], "answer": filtered_docs[0]["answer"]},
                          ensure_ascii=False)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Helper function to create a node for a given agent
def agent_node(state, agent):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and (last_message.name == "search_qa_records"):
        result = AIMessage(content=last_message.content)
        return {"messages": [result]}
    result = agent.invoke(state)
    return {
        "messages": [result],
    }


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                # "system",
                # "You are a helpful AI assistant, collaborating with other assistants."
                # " Use the provided tools to progress towards answering the question."
                # " If you are unable to fully answer, that's OK, another assistant with different tools "
                # " will help where you left off. Execute what you can to make progress."
                # " If you or any of the other assistants have the final answer or deliverable,"
                # " prefix your response with FINAL ANSWER so the team knows to stop."
                # " You have access to the following tools: {tool_names}.\n{system_message}",
                "system",
                "{system_message}"
                "You have access to the following tools: {tool_names}. "
                "Reject to answer if the query is not about problems of household cleaning machines."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)

system = """You can only help by searching similar historical problems. """

qa_searching_runnable = create_agent(
    llm,
    [search_qa_records],
    system_message=system,
)
search_qa_records_assistant = functools.partial(agent_node, agent=qa_searching_runnable)


# begin of graph
builder = StateGraph(State)

# node
builder.add_node("search_qa_records_assistant", search_qa_records_assistant)
builder.set_entry_point("search_qa_records_assistant")
builder.add_node(
    "search_qa_records_tools",
    create_tool_node_with_fallback([search_qa_records]),
)


# edge
def route_search_qa_recods(
    state: State,
) -> Literal[
    "search_qa_records_tools",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "search_qa_records_tools"


builder.add_edge("search_qa_records_tools", "search_qa_records_assistant")
builder.add_conditional_edges("search_qa_records_assistant", route_search_qa_recods)