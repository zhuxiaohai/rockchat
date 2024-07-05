from typing import List, Dict, Optional
from sqlalchemy.types import NullType
from sqlalchemy.schema import CreateTable
from sqlalchemy import (
    Table,
    select,
)
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase


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
    def get_table_info(self, table_names: Optional[List[str]] = None, table_selectors: [Dict] = {}) -> str:
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
                table_info += f"\n{self._get_sample_rows(table, selector)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_sample_rows(self, table: Table, selector: List = []) -> str:
        # build the select command
        if selector:
            columns_selected = [table.c[col.name] for col in table.columns if col.name in selector]
            command = select(*columns_selected).limit(self._sample_rows_in_table_info)
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