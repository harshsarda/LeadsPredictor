import pickle
import yaml
import os
import sys
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.lead_predictor import LeadPredictor
from src.logging_utils import get_logger

sys.path.append("./src/")


class ModelRequest(BaseModel):
    shift: List[float]
    gender: List[float]
    education: List[float]
    created_on: List[float]
    dow: List[float]
    employer_type: List[float]
    applicant_location: List[float]
    city: List[float]
    area: List[float]
    organization: List[float]
    deposit: List[float]
    category: List[float]
    english: List[float]
    num_openings: List[float]
    max_salary: List[float]
    min_salary: List[float]
    is_part_time: List[float]


logger = get_logger()

with open("./config/config.yaml", "r") as yaml_file:
    yaml_read = yaml.safe_load(yaml_file)

model_path = os.path.join(".", yaml_read["model_path"])
feature_transformers_path = os.path.join(".", yaml_read["feature_transformers_path"])
model_cols = yaml_read["model_cols"]

logger.info("num model cols: {}".format(len(model_cols)))
logger.info("model cols: {}".format("".join(model_cols)))

with (open(model_path, "rb")) as openfile:
    model_obj = pickle.load(openfile)

with (open(feature_transformers_path, "rb")) as openfile:
    feat_trans = pickle.load(openfile)

lead_predictor = LeadPredictor(
    feat_trans=feat_trans,
    model_cols=model_cols,
    models=model_obj["fitted_models"],
    logger=logger,
)

app = FastAPI()


@app.post("/items/")
async def get_leads_for_job_posting(item: ModelRequest):

    response_dict = lead_predictor.run(data_dict=item.dict())
    return response_dict
