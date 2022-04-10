import pandas as pd
import numpy as np
import os
import sys
import pickle

sys.path.append("./ml_auto/")

from data_utils import CatNumAgg, FreqEnc, gen_cat_cat
from custom_estimator import Estimator
from lightgbm import LGBMRegressor

DATA_DIR = "../data/"

df = pd.read_excel(os.path.join(DATA_DIR, "data.xlsx"), sheet_name=0)
print(df.shape)

df.drop("feebackgiven", axis=1, inplace=True)

df = df.sort_values("created_on").reset_index(drop=True)
df["target"] = np.log1p(df.leads_per_opening)

folds = [
    (
        df[(df.created_on <= df.created_on.quantile(0.7))].index.tolist(),
        df[
            (df.created_on > df.created_on.quantile(0.7))
            & (df.created_on <= df.created_on.quantile(0.8))
        ].index.tolist(),
    ),
    (
        df[
            (df.created_on >= df.created_on.quantile(0.1))
            & (df.created_on <= df.created_on.quantile(0.8))
        ].index.tolist(),
        df[
            (df.created_on > df.created_on.quantile(0.8))
            & (df.created_on <= df.created_on.quantile(0.9))
        ].index.tolist(),
    ),
    (
        df[
            (df.created_on >= df.created_on.quantile(0.2))
            & (df.created_on <= df.created_on.quantile(0.9))
        ].index.tolist(),
        df[
            (df.created_on > df.created_on.quantile(0.9))
            & (df.created_on <= df.created_on.quantile(1))
        ].index.tolist(),
    ),
]

print([(df.iloc[i].shape, df.iloc[j].shape) for i, j in folds])

target = df.target.values

df = gen_cat_cat(df)

cat_freq_cols = [
    "category",
    "gender",
    "organization",
]

fe = FreqEnc(cat_freq_cols=cat_freq_cols)
df = fe.fit_transform(df)

cat_num_agg_dict = {
    "area": {
        "applicant_location": ["std"],
        "education": ["std"],
        "num_openings": ["std"],
        "max_salary": ["median"],
    },
    "category_city": {
        "applicant_location": ["mean"],
        "max_salary": ["median"],
        "education": ["std"],
        "num_openings": ["std"],
    },
    "category": {
        "applicant_location": ["std"],
        "education": ["mean"],
    },
    "organization": {
        "applicant_location": ["std"],
        "min_salary": ["median", "std"],
    },
    "city": {
        "applicant_location": ["std"],
    },
    "category_dow": {
        "max_salary": ["mean"],
    },
}

catnumagg = CatNumAgg(cat_num_agg_dict=cat_num_agg_dict)
df = catnumagg.fit_transform(df)

print(df.shape)

model_cols = [
    "applicant_location",
    "applicant_location_mean_grpby_and_category_city",
    "applicant_location_std_grpby_and_area",
    "applicant_location_std_grpby_and_category",
    "applicant_location_std_grpby_and_city",
    "applicant_location_std_grpby_and_organization",
    "category_fe",
    "education",
    "education_mean_grpby_and_category",
    "education_std_grpby_and_area",
    "education_std_grpby_and_category_city",
    "english",
    "gender_fe",
    "max_salary",
    "max_salary_mean_grpby_and_category_dow",
    "max_salary_median_grpby_and_area",
    "max_salary_median_grpby_and_category_city",
    "min_salary_median_grpby_and_organization",
    "min_salary_std_grpby_and_organization",
    "num_openings",
    "num_openings_std_grpby_and_area",
    "num_openings_std_grpby_and_category_city",
    "organization_fe",
]

params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "min_child_samples": 90,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 16,
    "objective": "regression",
    "subsample": 1.0,
    "subsample_freq": 10,
}

est = Estimator(
    model=LGBMRegressor(**params),
    early_stopping_rounds=100,
    validation_scheme=folds,
    shuffle=True,
)

print(est.get_repeated_out_of_folds(df[model_cols].values, target))


feat_imps = est.feature_importances(columns=model_cols)
feat_imps["cum_imp"] = feat_imps.feature_importance.cumsum()
print(feat_imps)

est.save_model(file_name="../model/model.pkl")


with open("../feature_transformers/feat_trans.pkl", "wb") as out_file:
    pickle.dump({"fe": fe, "catnumagg": catnumagg}, out_file)
