from .process_data import DataProcessing
from .modeling_utils import get_predictions


class LeadPredictor:
    def __init__(self, feat_trans, model_cols, models, logger):
        self.data_processing = DataProcessing(
            **feat_trans, model_cols=model_cols, logger=logger
        )
        self.models = models
        self.logger = logger

    def get_response_dict(self, results):
        response_dict = {}
        response_dict["Payload"] = results
        response_dict["Total"] = len(results)
        return response_dict

    def run(self, data_dict):
        try:
            processed_data = self.data_processing.run(data_dict)
        except:
            self.logger.exception("data processing error")

        self.logger.info(
            "processed data shape: ({},{})".format(
                processed_data.shape[0], processed_data.shape[1]
            )
        )
        preds = get_predictions(self.models, processed_data)
        self.logger.info("preds shape: {}".format(preds.shape[0]))
        return self.get_response_dict(preds)
