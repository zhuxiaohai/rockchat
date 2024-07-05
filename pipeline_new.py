import getpass
import os
from find_keywords import LabellerByRulesWithPos
from pipeline import NL2SQLPineline
from vector_db import ChromaDB
from search_engine import QAVectorDBSearchEngine
from recall import RecallBySearchEngine
from nl2sql import SimpleLookup
from langchain_core.tools import tool
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import AzureChatOpenAI
from typing import Callable
from typing import Literal
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
#%%
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


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

#%%
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

pipeline_config = {
    "router": router_config,
    "recall": recall_config,
    "nl2sql": nl2sql_config,
}

nl2sql_pipeline = NL2SQLPineline(pipeline_config)
#%%
@tool
def fetch_table(request: str) -> str:
    """Consult a table about robot model parameters according to the query,
    and return the results."""
    results = nl2sql_pipeline.run(request)
    return results


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "fetch_table",
            ]
        ],
        update_dialog_stack,
    ]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""
    class Config:
        schema_extra = {
            "example 1": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 2": {
                "cancel": False,
                "reason": "I need to get more information about robot model names.",
            },
        }
    cancel: bool = Field(description="If the current task should be canceled.")
    reason: str = Field(description="reason for cancelling or not")
#%%
# Table Searching Assistant
table_searching_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for searching table about robot model parameters."
            "The primary assistant delegates work to you whenever the user needs to know parameters about robot models."
            "The query must have robot model names such as A10, G20, H1 and so on to complete the search, and"
            ' if it does not have that information, use the tool "CompleteOrEscalate" to escalate the task back to the main assistant, '
            " but remember to set cancel = False."
            'If you have successfully used the "fetch_table" tool, please use "CompleteOrEscalate" tool to escalate the task back to the main assistant.'
            "\n\nIf none of your tools are appropriate, then use the tool "
            ' "CompleteOrEscalate" to send the dialog back to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "directly use the user query as input to your tools, and don't make any translations or rephrases.",
        ),
        ("placeholder", "{messages}"),
    ]
)
llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)
table_searching_runnable = table_searching_prompt | llm.bind_tools(
     [CompleteOrEscalate, fetch_table]
)
#%%
# Primary Assistant
class ToTableSearchingAssistant(BaseModel):
    """Transfers work to a specialized assistant to search table for robot model parameters."""
    request: str = Field(
        description="directly use the user query as input to the specialized assistant, and don't make any translations or rephrases."
    )
llm_p = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for home cleaning robot. "
            "Your primary role is to answer customer queries about the home cleaning robot. "
            "If a customer requests you to answer some questions outof the scope just reject to answer."
            "If a customer requests to know some parameter of a robot, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
        ),
        ("placeholder", "{messages}"),
    ]
)
assistant_runnable = primary_assistant_prompt | llm_p.bind_tools(
    [
        ToTableSearchingAssistant
    ]
)


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        request = state["messages"][-1].tool_calls[0]["args"]["request"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the task is not complete until after you have successfully invoked the appropriate tool."       
                    " If the you needs more information to complete the task, call the CompleteOrEscalate function to let the primary host assistant take control."
                    "If you successfullyt complete the task, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant."
                    "Directly use the request below as input to your tools and don't translate into English or rephase."
                    f"\n\nrequest: {request}",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node
#%%
builder = StateGraph(State)
#%%
# fetch table assistant
builder.add_node(
    "enter_search_table",
    create_entry_node("Assistant of Searching Robot Model Parameters", "fetch_table"),
)
builder.add_node("fetch_table", Assistant(table_searching_runnable))
builder.add_edge("enter_search_table", "fetch_table")
builder.add_node(
    "fetch_table_tools",
    create_tool_node_with_fallback([fetch_table]),
)


def route_fetch_table(
    state: State,
) -> Literal[
    "fetch_table_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "fetch_table_tools"


builder.add_edge("fetch_table_tools", "fetch_table")
builder.add_conditional_edges("fetch_table", route_fetch_table)
#%%
# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")
#%%
# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.set_entry_point("primary_assistant")

def route_primary_assistant(
    state: State,
) -> Literal[
    "enter_search_table",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToTableSearchingAssistant.__name__:
            return "enter_search_table"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_search_table": "enter_search_table",
        END: END,
    },
)
# # Each delegated workflow can directly respond to the user
# # When the user responds, we want to return to the currently active workflow
# def route_to_workflow(
#     state: State,
# ) -> Literal[
#     "fetch_table",
# ]:
#     """If we are in a delegated state, route directly to the appropriate assistant."""
#     dialog_state = state.get("dialog_state")
#     if not dialog_state:
#         return "primary_assistant"
#     return dialog_state[-1]


# builder.add_conditional_edges("primary_assistant", route_to_workflow)
#%%
# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
part_4_graph = builder.compile(
    checkpointer=memory,
)
#%%
from IPython.display import Image, display

try:
    display(Image(part_4_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


import uuid
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight informatio
        # Checkpoints are accessed by thread_id
        "thread_id": "198197a5-c74c-463d-8312-ded3f1d94479",
    }
}
#%%
tutorial_questions = [
    "你好",
    "G20电源线多长.",
    "和1米比哪个长",
    "帮我查询下天气好吗",
]
#%%
_printed = set()
for question in tutorial_questions:
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)