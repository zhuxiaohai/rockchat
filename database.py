import pandas as pd
def format_model(x):
    model_list = x.split(',')
    model_list = [i.strip().lower().replace(" ", "") for i in model_list]
    new_list = [model_list[0]]
    i = 1
    while i < len(model_list):
        if (i != len(model_list) - 1) and (model_list[i-1] == model_list[i]):
            new_list.append(model_list[i]+model_list[i+1])
            if i < len(model_list) - 1:
                i += 2
            else:
                break
        elif (i != len(model_list) - 1) and (model_list[i-1] != model_list[i]):
            new_list.append(model_list[i])
            i += 1
        elif (model_list[i] == "上下水") or (model_list[i] == "air"):
            for j in range(len(new_list)):
                if model_list[i-1] == new_list[j]:
                    new_list.pop(j)
                    break
            new_list.append(model_list[i-1]+model_list[i])
            i += 1
        else:
            new_list.append(model_list[i])
            break
    return new_list

class DataQuery:
    def __init__(self):
        self.sweeping_path = "/data/dataset/kefu/sweeping.csv"
        self.mopping_path = "/data/dataset/kefu/mopping.csv"
        self.washing_path = "/data/dataset/kefu/washing.csv"
        self.dim_path = "/data/dataset/kefu/dim_df20240315.csv"

        self.sweeping_df = pd.read_csv(
            self.sweeping_path)
        self.mopping_df = pd.read_csv(
            self.mopping_path)
        self.washing_df = pd.read_csv(
            self.washing_path)
        self.dim_df = pd.read_csv(
            self.dim_path)
        # self.all_data = {
        #     "扫地机": self.sweeping_df,
        #     "洗地机": self.mopping_df,
        #     "洗衣机": self.washing_df,
        # }
        self.all_data = pd.concat([self.sweeping_df, self.mopping_df, self.washing_df], axis=0)
        self.all_data["商品型号"] = self.all_data["商品型号"].apply(lambda x: format_model(x)[0])

    # def query_data(self, model):
    #     category = self.dim_df.loc[self.dim_df["model"] == model, "cat_name"].iloc[0]
    #     return self.all_data[category]

    def query_data(self, model):
        return self.all_data
