from typing import Annotated, Optional, Literal, Callable
from typing_extensions import TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

from langchain_openai import AzureChatOpenAI

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from table_agent import builder as table_agent_builder

from qa_agent import builder as qa_agent_builder


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
                "PrimaryAssistant",
                "QATeam",
                "SearchTableTeam",
            ]
        ],
        update_dialog_stack,
    ]


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


def route_primary_assistant(
    state: State,
) -> Literal[
    "enter_SearchTableTeam",
    "enter_QATeam",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToTableSearchingAssistant.__name__:
            return "enter_SearchTableTeam"
        elif tool_calls[0]["name"] == ToQARecordsAssistant.__name__:
            return "enter_QATeam"
    raise ValueError("Invalid route")


def route_qa_team(
    state: State,
) -> Literal[
    "leave_assistant",
    "leave_assistant_end",
]:
    last_message = state["messages"][-1]
    if last_message.content == "搜索结果和问题的匹配度很低，请换个问法":
        status = "ask_for_human"
    else:
        status = "reset"
    if status == "ask_for_human":
        return "leave_assistant"
    else:
        return "leave_assistant_end"


def route_search_table_team(
    state: State,
) -> Literal[
    "leave_assistant",
    "leave_assistant_end",
    "__end__",
]:
    last_message = state["messages"][-1]
    if last_message.content == "问题中需要提供型号名称才能完成搜索":
        status = "retry"
    elif last_message.content == "搜索结果和问题的匹配度很低，请换个问法":
        status = "ask_for_human"
    else:
        status = "reset"
    if status == "retry":
        return "__end__"
    elif status == "ask_for_human":
        return "leave_assistant"
    else:
        return "leave_assistant_end"


def set_entry_point(
    state: State,
) -> Literal[
    "PrimaryAssistant",
    "SearchTableTeam",
    "QATeam",
]:
    current_at = state.get("dialog_state", [])
    if not current_at:
        return "PrimaryAssistant"
    else:
        return current_at[-1]


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
        return {"messages": [result]}


class ToTableSearchingAssistant(BaseModel):
    """Transfers work to a specialized assistant capable of searching table about parameter or features
 of cleaning machines."""
    query: str = Field(
        description="Ensure to keep necessary information reflecting on above conversations "
                    "and keep the language the same as the original."
    )


class ToQARecordsAssistant(BaseModel):
    """Transfers work to a specialized assistant capable of solving problems or
 looking up similar historical cases in usage of household cleaning machines."""
    query: str = Field(
        description="Ensure to keep necessary information reflecting on above conversations "
                    "and keep the language the same as the original."
    )


llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://csagent.openai.azure.com/",
    api_key="346ac6661e314a9d8b91b6a99202ba42",
    deployment_name="gpt-4-8k",
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for household cleaning machines. "
            "Reject to answer if a customer queries about something out of this scope. "
            "When a task takes expertise, delegate it to the appropriate specialized assistant"
            " by invoking the corresponding tool. "
            "You have access to the following tools: {tool_names}. "
            'Invoke the tool "ToTableSearchingAssistant" and reroute the human problem to it'
            ' once you receive a message "搜索结果和问题的匹配度很低，请换个问法" from "QATeam".',
        ),
        ("placeholder", "{messages}"),
    ]
)
primary_assistant_prompt = primary_assistant_prompt.partial(tool_names=", ".join([
    ToQARecordsAssistant.__name__,
    ToTableSearchingAssistant.__name__,
]))
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    [
        ToQARecordsAssistant, ToTableSearchingAssistant
    ]
)


def create_entry_node(assistant_name: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        tool_call_name = state["messages"][-1].tool_calls[0]["name"]
        query = state["messages"][-1].tool_calls[0]["args"]["query"]
        return {
            "messages": [
                ToolMessage(
                    content=f"{assistant_name} is delegated to work as proxy."
                    f"\n\nquery: {query}",
                    name=tool_call_name,
                    tool_call_id=tool_call_id,
                ),

            ],
            "dialog_state": assistant_name,
        }
    return entry_node


def get_last_messages(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and (last_message.name in [
        ToTableSearchingAssistant.__name__,
        ToQARecordsAssistant.__name__
    ]):
        messages = [HumanMessage(content=last_message.content)]
    # sender = state.get("sender", None)
    # messages = []
    # if sender == "PrimaryAssistant":
    #     for message in state["messages"][::-1]:
    #         if isinstance(message, HumanMessage):
    #             messages = [HumanMessage(content=message.content)]
    #             break
    # else:
    else:
        messages = state["messages"]
    return {"messages": messages}


def join_graph(response: dict, name):
    return {"messages": [AIMessage(response["messages"][-1].content, name=name)]}


# begin of graph
builder = StateGraph(State)

# search_table subgraph
search_table_subgraph = table_agent_builder.compile()
# search_table_subgraph = enter_chain | search_table_subgraph
search_table_chain = get_last_messages | search_table_subgraph | functools.partial(join_graph, name="SearchTableTeam")
builder.add_node("SearchTableTeam", search_table_chain)
builder.add_conditional_edges(
    "SearchTableTeam",
    route_search_table_team,
)

# search_qa_records subgraph
qa_agent_subgraph = qa_agent_builder.compile()
# qa_agent_subgraph = enter_chain | qa_agent_subgraph
qa_agent_chain = get_last_messages | qa_agent_subgraph | functools.partial(join_graph, name="QATeam")
builder.add_node("QATeam", qa_agent_chain)
builder.add_conditional_edges(
    "QATeam",
    route_qa_team,
)

# primary_assistant graph
# node
builder.add_node("PrimaryAssistant", Assistant(assistant_runnable))
builder.set_conditional_entry_point(
    set_entry_point,
)
builder.add_node(
    "enter_QATeam",
    create_entry_node("QATeam"),
)
builder.add_node(
    "enter_SearchTableTeam",
    create_entry_node("SearchTableTeam"),
)
builder.add_node("leave_assistant", pop_dialog_state)
builder.add_node("leave_assistant_end", pop_dialog_state)

# edge
builder.add_conditional_edges(
    "PrimaryAssistant",
    route_primary_assistant,
)
builder.add_edge("enter_QATeam", "QATeam")
builder.add_edge("enter_SearchTableTeam", "SearchTableTeam")
builder.add_edge("leave_assistant", "PrimaryAssistant")
builder.add_edge("leave_assistant_end", END)

# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
part_4_graph = builder.compile(
    checkpointer=memory,
)
