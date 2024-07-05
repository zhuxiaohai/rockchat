from abc import ABC, abstractmethod
import pandas as pd
from database import SQLAlchemyDB


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    # Get the headers
    headers = df.columns.tolist()

    # Create the header row
    header_row = '| ' + ' | '.join(headers) + ' |'

    # Create the separator row
    separator_row = '| ' + ' | '.join(['---'] * len(headers)) + ' |'

    # Create the data rows
    data_rows = []
    for _, row in df.iterrows():
        data_rows.append('| ' + ' | '.join(map(str, row.tolist())) + ' |')

    # Combine all rows into a single markdown table string
    markdown_table = '\n'.join([header_row, separator_row] + data_rows)

    return markdown_table


class Tabler(ABC):
    def __init__(self, config):
        db = SQLAlchemyDB(config["db_absolute_path"])
        if config["reset"]:
            df = pd.read_csv(config["df_path"])
            db.build_table(config["table_name"], df)
        self.db = db
        self.table_name = config["table_name"]
        self.output_type = config.get("output_type", "dataframe")

    @abstractmethod
    def prepare_sql(self, query, recall):
        pass

    def predict(self, query, recall):
        sql = self.prepare_sql(query, recall)
        result_df = pd.read_sql_query(sql, self.db.engine)
        if self.output_type == "markdown":
            result_df = dataframe_to_markdown(result_df)
        return result_df


class SimpleLookup(Tabler):
    def prepare_sql(self, query, recall):
        primary_key = ["商品型号"]
        cat_list = [cat["word"] == "sweeping" for cat in query["labels"]["cat"]]
        if any(cat_list):
            primary_key += ["版本"]

        keyword_list = sorted(recall, key=lambda x: x["entity"]["start"])
        keyword_list = [item["page_content"] for item in keyword_list]

        model_list = query["labels"]["model"]
        model_with_no_version = sorted([model for model in model_list if model["word"].find("上下水") < 0],
                                       key=lambda x: x["start"])
        model_with_no_version = [model["word"] for model in model_with_no_version]
        model_with_version = sorted([model for model in model_list if model["word"].find("上下水") >= 0],
                                    key=lambda x: x["start"])
        model_with_version = [model["word"].replace("上下水", "").replace("版本", "").replace("版", "").replace(" ", "")
                              for model in model_with_version]
        where_condition = f"""
        WHERE `商品型号` IN ({', '.join(f"'{x}'" for x in model_with_no_version)})
        OR (`商品型号` IN ({', '.join(f"'{x}'" for x in model_with_version)}) AND `版本` = '上下水版')
        """.lstrip()

        sql = f"""
        SELECT 
            {', '.join(f"{x}" for x in primary_key+keyword_list)}
        FROM
            {self.table_name}
        {where_condition}
        """

        return sql