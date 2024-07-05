from nl2sql import SimpleLookup


def test_simple_lookup():
    config = {
        "db_absolute_path": "/root/PycharmProjects/rockchat/data/model_params.db",
        "reset": False,
        "df_path": "/root/PycharmProjects/rockchat/data/model_params20240620.csv",
        "table_name": "model_params20240620",
    }
    tabler = SimpleLookup(config)

    query = {"query": "G20的电源线有多长",
             "labels": {
                 "model": [{"word": "g20", "start": 0, "end": 3}],
                 "cat": [{"word": "sweeping", "start": 0, "end": 3}]
             }
             }
    recall = [{"entity": {"word": "电源线", "start": 10, "end": 14}, "page_content": "电源线长"}]
    result = tabler.predict(query, recall)
    result = result.to_dict(orient="records")
    assert len(result) == 2
    assert result[0]["电源线长"] == "180cm"
    assert result[1]["电源线长"] == "180cm"