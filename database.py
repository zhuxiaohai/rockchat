import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


class SQLAlchemyDB:
    def __init__(self, db_absolute_path):
        self.engine = create_engine(f"sqlite:///{db_absolute_path}", connect_args={'timeout': 15})
        self.cursor = self.engine.connect()

    def close(self):
        self.cursor.close()

    def reset(self):
        all_tables = self.cursor.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        all_tables = [table[0] for table in all_tables]
        for table in all_tables:
            self.cursor.exec_driver_sql(f"DROP TABLE IF EXISTS {table};")

    def delete_table(self, table):
        all_tables = self.cursor.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        all_tables = [table[0] for table in all_tables]
        if table in all_tables:
            self.cursor.exec_driver_sql(f"DROP TABLE IF EXISTS {table};")

    def build_table(self, table_name, df):
        self.delete_table(table_name)
        dtype_mapping = {col: sqlalchemy.types.String for col in df.columns}
        df.to_sql(table_name, self.engine, if_exists="replace", index=False, dtype=dtype_mapping)


if __name__ == "__main__":
    db_absolute_path = "/root/PycharmProjects/rockchat/data/model_params.db"
    db = SQLAlchemyDB(db_absolute_path)
    df_path = "/root/PycharmProjects/rockchat/data/model_params20240620.csv"
    df = pd.read_csv(df_path)
    table_name = "model_params20240620"
    db.build_table(table_name, df)
    db.close()