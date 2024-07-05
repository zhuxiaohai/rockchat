from typing import Annotated, Literal
from typing_extensions import TypedDict
import functools

from find_keywords import LabellerByRulesWithPos
from vector_db import ChromaDB
from search_engine import QAVectorDBSearchEngine
from recall import RecallBySearchEngine
from nl2sql import SimpleLookup

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


router_config = {
    "class": LabellerByRulesWithPos,
    "config": {
        "dim_df_path": "data/dim_df20240619.csv",
        "model_col": ("model", "model"),
        "cat_col": ("cat_cn", "cat"),
        "error_col": ("error", "error"),
        "ner_model_path": "/workspace/data/private/zhuxiaohai/models/bert_finetuned_ner_augmented/"
    }
}

recall_config = {
    "class": RecallBySearchEngine,
    "config": {
        "search_engine": {
            "class": QAVectorDBSearchEngine,
            "docs_path": "data/table_entity.csv",
            "docs_col": "entity_name",
            "ids_col": "ids",
            "metadata_cols": ["sweeping", "washing", "mopping", "content"],
            "collection_name": "table_entity",
            "reset": False,
            "encoder_path": "/workspace/data/private/zhuxiaohai/models/bge_finetuned_emb_ner_link",
            "vector_db": {
                "class": ChromaDB,
                "host": "/data/dataset/kefu/chroma",
            },
        },
    },
    "top_n": 1,
}

nl2sql_config = {
    "class": SimpleLookup,
    "config": {
        "db_absolute_path": "/root/PycharmProjects/rockchat/data/model_params.db",
        "reset": False,
        "df_path": "/root/PycharmProjects/rockchat/data/model_params20240620.csv",
        "table_name": "model_params20240620",
        "output_type": "markdown"
    }
}

extractor = router_config["class"](router_config["config"])
recall = recall_config["class"](recall_config["config"])
nl2sql = nl2sql_config["class"](nl2sql_config["config"])


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
def search_table(question):
    """Consult a table about parameters or features of household cleaning machines according to the question,
 and return the results."""
    query_body = extractor.extract_keywords(question)
    if not query_body["labels"]["model"]:
        return "问题中需要提供型号名称才能完成搜索"
    query_body.update({"top_n": recall_config["top_n"]})
    recalled_cols = recall.query_recalls(query_body)
    pairs = [{"question": question, "document": d['page_content']} for d in recalled_cols]
    scores = retrieval_grader.batch(pairs)
    filtered_docs = [recalled_cols[i] for i in range(len(scores)) if scores[i].binary_score == "yes"]
    if not filtered_docs:
        return "搜索结果和问题的匹配度很低，请换个问法"
    else:
        tabler_results = nl2sql.predict(query_body, filtered_docs)
        return tabler_results


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


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
                "You have access to the following tools: {tool_names}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# Helper function to create a node for a given agent
def agent_node(state, agent):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and (last_message.name == "search_table"):
        result = AIMessage(content=last_message.content)
        return {"messages": [result]}
    result = agent.invoke(state)
    return {
        "messages": [result],
    }


llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)


system = """You are a specialized assistant for searching table about parameters or features \
of cleaning machines. Use the language the same as the query to answer. \
If the query is not about cleaning machines, just reject to answer, otherwise always invoke your tools. """
# Research agent and node
table_searching_runnable = create_agent(
    llm,
    [search_table],
    system_message=system,
)
search_table_assistant = functools.partial(agent_node, agent=table_searching_runnable)


# begin of graph
builder = StateGraph(AgentState)

# node
builder.add_node("search_table_assistant", search_table_assistant)
builder.set_entry_point("search_table_assistant")
builder.add_node(
    "search_table_tools",
    create_tool_node_with_fallback([search_table]),
)


# edge
def route_search_table(
    state: AgentState,
) -> Literal[
    "search_table_tools",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "search_table_tools"


builder.add_edge("search_table_tools", "search_table_assistant")
builder.add_conditional_edges("search_table_assistant", route_search_table)