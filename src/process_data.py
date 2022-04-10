import pandas as pd

from .data_utils import gen_cat_cat


class DataProcessing:
    def __init__(self, fe, catnumagg, model_cols, logger):
        """
        FreqEnc object
        OHE object
        CatNumAgg object
        model_cols: List of columns to be used for the model
        """
        self.fe = fe
        self.catnumagg = catnumagg
        self.model_cols = model_cols
        self.logger = logger

    def run(self, data_dict):
        df = pd.DataFrame(data_dict)
        # generate derived categories
        df = gen_cat_cat(df)
        self.logger.info("data shape after cat addition: {}".format(df.shape[1]))
        # perform frequency encoding
        df = self.fe.transform(df)
        self.logger.info("data shape after freq enc: {}".format(df.shape[1]))
        # perform cat num aggregations
        df = self.catnumagg.transform(df)
        self.logger.info("data shape after catnumagg: {}".format(df.shape[1]))
        # subset model cols
        df = df[self.model_cols].values
        self.logger.info("final data shape: ({},{})".format(df.shape[0], df.shape[1]))
        return df
