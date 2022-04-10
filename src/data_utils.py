import pandas as pd


def get_derived_cat(df, a, b):
    df[a + "_" + b] = df[a].astype("str") + "_" + df[b].astype("str")
    return df


def gen_cat_cat(df):
    df = get_derived_cat(df, "category", "city")
    df = get_derived_cat(df, "category", "dow")
    return df


class FreqEnc:
    def __init__(self, cat_freq_cols):
        self.cat_freq_cols = cat_freq_cols

    def fit(self, df):
        self.freq_encoding_dict = {
            x: df[x].value_counts(1).to_dict() for x in self.cat_freq_cols
        }

    def transform(self, df):
        for col in self.cat_freq_cols:
            df[col + "_fe"] = df[col].map(self.freq_encoding_dict[col]).fillna(0)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

class CatNumAgg:
    def __init__(self, cat_num_agg_dict):
        self.cat_num_agg_dict = cat_num_agg_dict

    def fit(self, df):
        self.encoding_dict = {}
        for grp_col, agg_dict in self.cat_num_agg_dict.items():
            li = []
            cols = [grp_col]
            for agg_col, agg_funcs in agg_dict.items():
                agg_df = df.groupby(grp_col)[agg_col].agg(agg_funcs)
                cols.extend(
                    [agg_col + "_" + j + "_grpby_and_" + grp_col for j in agg_funcs]
                )
                li.append(agg_df)
            final_df = pd.concat(li, axis=1).reset_index()
            final_df.columns = cols
            self.encoding_dict[grp_col] = final_df.to_dict()

    def transform(self, df):
        for k in self.encoding_dict:
            agg_df = pd.DataFrame(self.encoding_dict[k])
            df = df.merge(agg_df, on=k, how="left")
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
