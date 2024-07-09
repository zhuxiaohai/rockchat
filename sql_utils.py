from typing import List, Dict, Optional, Union, Any
from sqlalchemy.types import NullType
from sqlalchemy.schema import CreateTable
from sqlalchemy import (
    Table,
    select,
    and_,
)
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import SQLInput, SQLInputWithTables, _strip
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough


class CreateTableWithSelector(CreateTable):
    """Represent a CREATE TABLE statement."""
    def __init__(
        self,
        element,
        include_foreign_key_constraints=None,
        if_not_exists=False,
        selector=[],
    ):
        super().__init__(
            element=element,
            include_foreign_key_constraints=include_foreign_key_constraints,
            if_not_exists=if_not_exists
        )
        if selector:
            self.columns = [sqlalchemy_col for sqlalchemy_col, column
                            in zip(self.columns, element.columns)
                            if column.name in selector]


class SQLDatabaseWithSelector(SQLDatabase):
    def get_table_info(self, table_names: Optional[List[str]] = None,
                       table_selectors: Optional[Dict] = {},
                       labels: Optional[Dict] = {},
                       ) -> str:
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            self._metadata.reflect(
                views=self._view_support,
                bind=self._engine,
                only=list(to_reflect),
                schema=self._schema,
            )

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            selector = table_selectors.get(table.name, [])
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # Ignore JSON datatyped columns
            for k, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)

            # add create table command
            create_table = str(CreateTableWithSelector(table, selector=selector).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            has_extra_info = (
                    self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table, selector, labels)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_sample_rows(self, table: Table, selector: List = [], labels: Optional[Dict] = {}) -> str:
        # build the select command
        if selector:
            limit = max(self._sample_rows_in_table_info, 3, len(labels.get("model", [])))
            condition = []
            for col in table.columns:
                if (col.name in selector) and (col.name in ["商品型号", "版本", "商品分类"]):
                    condition.append(table.c[col.name].isnot(None))
            columns_selected = [table.c[col.name] for col in table.columns if col.name in selector]
            if condition:
                command = select(*columns_selected).where(and_(*condition)).limit(limit)
            else:
                command = select(*columns_selected).limit(limit)
        else:
            command = select(table).limit(self._sample_rows_in_table_info)

        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns if col.name in selector])

        try:
            # get the sample rows
            with self._engine.connect() as connection:
                sample_rows_result = connection.execute(command)  # type: ignore
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )

            columns_str_list = columns_str.split("\t")
            if ("商品型号" in columns_str_list) or ("版本" in columns_str_list) or ("商品分类" in columns_str_list):
                model_list = [model["word"] for model in labels.get("model", [])]
                version_list = ["标准版", "上下水版"]
                cat_list = ["扫地机", "洗地机", "洗衣机"]
                num_model = len(model_list)
                num_versions = len(version_list)
                num_cat = len(cat_list)
                for r_id, row in enumerate(sample_rows):
                    if (r_id < num_versions) or (r_id < num_cat) or (r_id < num_model):
                        new_row = []
                        for col, value in zip(columns_str_list, row):
                            if (col == "版本") and (r_id < num_versions):
                                new_row.append(version_list[r_id])
                            elif (col == "商品分类") and (r_id < num_cat):
                                new_row.append(cat_list[r_id])
                            elif (col == "商品型号") and (r_id < num_model):
                                new_row.append(model_list[r_id])
                            else:
                                new_row.append(value)
                        sample_rows[r_id] = new_row

            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )


def create_sql_query_chain_with_selector(
    llm: BaseLanguageModel,
    db: SQLDatabaseWithSelector,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]:
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info"}.difference(prompt_to_use.input_variables):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use"),
            table_selectors=x.get("table_selectors", {}),
            labels=x.get("labels", {}),
        ),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )