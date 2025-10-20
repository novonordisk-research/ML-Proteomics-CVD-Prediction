import os
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
import pandas as pd
import numpy as np

sys.path.append("../ukbb_preprocessing/")

from raw_data_preprocessing.raw_data_loader import raw_data_loader
from raw_data_preprocessing.constants import *


def rename_variables(df):

    df = df.copy()
    original_columns = set(df.columns)

    df = df.set_index('IID')

    # rename variables
    df["study_date"] = df["p53_i0"].astype('datetime64[ns]')
    df["age"] = df["p21003_i0"]

    df["sex"] = df["p31"] # 0. female; 1. male
    df["smoking"] = df["p20116_i0"]
    df["alcohol_intake"] = df["p1558_i0"]
    df["bmi"] = df["p21001_i0"]
    df["waist_circumference"] = df["p48_i0"]
    df["hip_circumference"] = df["p49_i0"]
    df["diabetes"] = df["p2443_i0"]
    # cardio phenotypes
    df["systolic_blood_pressure"] = (df["p4080_i0_a0"] + df["p4080_i0_a1"]) / 2
    df["systolic_blood_pressure"] = df["systolic_blood_pressure"].fillna(df["p4080_i0_a0"])
    df["systolic_blood_pressure"] = df["systolic_blood_pressure"].fillna(df["p4080_i0_a1"])
    df["diastolic_blood_pressure"] = (df["p4079_i0_a0"] + df["p4079_i0_a1"]) / 2
    df["diastolic_blood_pressure"] = df["diastolic_blood_pressure"].fillna(df["p4079_i0_a0"])
    df["diastolic_blood_pressure"] = df["diastolic_blood_pressure"].fillna(df["p4079_i0_a1"])
    # diseases
    # 1. Heart attack; 2. Angina; 3. Stroke; 4. High blood pressure
    # -7. None of the above; -3. Prefer not to answer
    df["self_reported_cvd"] = df["p6150_i0"]

    # df["icd10_diagnoses_date"] = 41280

    # compare each column in icd10_dates to the baseline
    def censor_phenotypes(phenotypes, dates, baseline):
        
        phenotypes = phenotypes.str.split('|').values.tolist()
        dates = dates.apply(pd.to_datetime, errors='coerce')

        masks = []

        for i, col in enumerate(dates.columns):
            mask = dates[col].astype('datetime64[ns]') > baseline
            masks.append(mask)

        masks = pd.concat(masks, axis=1).values

        n_dates = max([len(x) if x is not np.nan else 0 for x in phenotypes])

        # expand icd10 list of lists into a matrix
        pheno_matrix = np.full(masks.shape, "None")
        for i, row in enumerate(phenotypes):
            if row is not np.nan:
                pheno_matrix[i, :len(row)] = row

        pheno_matrix[masks] = ""
        pheno_matrix[pheno_matrix == "None"] = ""
        return [list(x[x != ""]) for x in pheno_matrix]

    baseline = df['p53_i0'].astype('datetime64[ns]')

    df["icd10_diagnoses"] = censor_phenotypes(df["p41270"], df.filter(regex=("p41280_a*")), baseline)
    df["icd9_diagnoses"] = censor_phenotypes(df["p41271"], df.filter(regex=("p41281_a*")), baseline)

    df['medications'] = df['p20003_i0'].str.split('|')
    df['medications'] = df['medications'].fillna("").apply(list)
    # 1. Cholesterol lowering; 2. Blood pressure; 3. Insulin; # 4. Hormone replacement therapy; 
    # 5. Oral contraceptive pill or minipill # -7. None of the above; -1. Do not know;  # -3. Prefer not to answer
    df["cvd_meds_f"] = df["p6153_i0"].str.split('|')
    # 1. Cholesterol lowering; 2. Blood pressure; 3. Insulin; # -7. None of the above; -1. Do not know; 
    # -3. Prefer not to answer
    df["cvd_meds_m"] = df["p6177_i0"].str.split('|')
    df["cvd_meds"] = np.where(df["sex"] == 0, df["cvd_meds_f"], df["cvd_meds_m"])
    # blood markers
    df["tc"] = df["p30690_i0"]
    df["ldlc"] = df["p30780_i0"]
    df["hdlc"] = df["p30760_i0"]
    df["tg"] = df["p30870_i0"]
    df["glu"] = df["p30740_i0"]
    df["hbA1c"] = df["p30750_i0"]
    df["ast"] = df["p30650_i0"]
    df["alt"] = df["p30620_i0"]
    df["crp"] = df["p30710_i0"]
    df["rati"] = df["tc"] / df["hdlc"]
    
    # urine markers
    # df["u_albumin"] = df["p30500_i0"]
    # df["u_creatinine"] = df["p30510_i0"]

    # eGFR
    # From Table 2 at https://www.nejm.org/doi/full/10.1056/NEJMoa2102953
    # μ × min(Scr/κ,1)^a1 × max(Scr/κ,1)^a2 × min(Scys/0.8,1)^b1 ×
    # max(Scys/0.8,1)^b2 × c^Age × d[if female] × e[if Black]

    # Creatinine-based eGFR
    # convert umol/L to mg/dL
    df["creatinine"] = df["p30700_i0"] * 0.0113

    mu = 142
    k = np.where(df["sex"] == 0, 0.7, 0.9)
    a1 = np.where(df["sex"] == 0, -0.241, -0.302)
    a2 = -1.200
    c = 0.993
    d = np.where(df["sex"] == 0, 1.012, 0)

    df["egfr_creat"] = mu * \
                       np.minimum(df["creatinine"] / k, 1) ** a1 * \
                       np.maximum(df["creatinine"] / k, 1) ** a2 * \
                       c ** df["age"] + \
                       d

    # Cystatin C + creatinine-based eGFR
    df["cystatin_c"] = df["p30720_i0"] # mg/L

    mu = 135
    a1 = np.where(df["sex"] == 0, -0.299, -0.144)
    a2 = -0.544
    b1 = -0.323
    b2 = -0.778
    c = 0.9961
    d = np.where(df["sex"] == 0, 0.963, 0)

    df["egfr_creat_cys"] = mu * \
                           np.minimum(df["creatinine"] / k, 1) ** a1 * \
                           np.maximum(df["creatinine"] / k, 1) ** a2 * \
                           np.minimum(df["cystatin_c"] / 0.8, 1) ** b1 * \
                           np.maximum(df["cystatin_c"] / 0.8, 1) ** b2 * \
                           c ** df["age"] + \
                           d

    # socioeconomics
    df["income"] = df["p738_i0"]
    df["sr-ethnicity"] = df["p21000_i0"]
    df["home_owner"] = df["p680_i0"]
    df["private_health"] = df["p4674_i0"]
    df["qualifications"] = df["p6138_i0"].str.split('|')
    df["tdi"] = df["p22189"]

    # family history
    df["sibling_illnesses"] = df["p20111_i0"].str.split('|')
    df["mother_illnesses"] = df["p20110_i0"].str.split('|')
    df["father_illnesses"] = df["p20107_i0"].str.split('|')

    # prss
    df["prs_cvd"] = df["p26223"]
    df["eprs_cvd"] = df["p26224"]
    df["prs_cad"] = df["p26227"]
    df["eprs_cad"] = df["p26228"]
    df["prs_ht"] = df["p26244"]
    df["eprs_ht"] = df["p26245"]
    df["prs_iss"] = df["p26248"]
    df["eprs_iss"] = df["p26249"]

    # compute time since first event
    cvd_event_dates = ["p131056", "p131270", "p131272", "p131274", "p131276", "p131278", "p131280", "p131282",
                    "p131284", "p131286", "p131288", "p131290", "p131292", "p131294", "p131296", "p131298",
                    "p131300", "p131302", "p131304", "p131306", "p131308", "p131310", "p131312", "p131314",
                    "p131316", "p131318", "p131320", "p131322", "p131324", "p131326", "p131328", "p131330",
                    "p131332", "p131334", "p131336", "p131338", "p131340", "p131342", "p131344", "p131346",
                    "p131348", "p131350", "p131352", "p131354", "p131356", "p131358", "p131360", "p131362",
                    "p131364", "p131366", "p131368", "p131370", "p131372", "p131374", "p131376", "p131378",
                    "p131380", "p131382", "p131384", "p131386", "p131388", "p131390", "p131392", "p131394",
                    "p131396", "p131398", "p131400", "p131402", "p131404", "p131406", "p131408", "p131410",
                    "p131412", "p131414", "p131416", "p131418", "p131420", "p131422"]

    dates_df = df[cvd_event_dates].copy()

    for col in dates_df.columns:
        dates_df[col] = pd.to_datetime(dates_df[col], errors='coerce')

    cvd_event_date = dates_df.min(axis=1)

    diff_years = (baseline - cvd_event_date)
    diff_years = diff_years.dt.days / 365.25

    df["years_since_first_event"] = diff_years
    
    # save the new columns
    new_columns = set(df.columns)

    # output_columns = ["IID"]
    output_columns = list(new_columns - original_columns)
    df = df[output_columns]

    return df


def save_results(scores, path):

    loader = raw_data_loader()

    os.makedirs(path, exist_ok=True)
    
    scores.to_csv(f'{path}/all_participants.csv')

    splits = loader._load_csv_folder(data_asset_name="ukbb_golden_splits_by_diagnosis_date", version=3)
    splits = [split.set_index(split.IID.astype(int)) for split in splits]

    for i, s in enumerate(splits):
        s["IID"] = s["IID"].astype(int)
        s = s.set_index("IID").merge(scores, on="IID")
        s.to_csv(f'{path}/npx_clin_ascvd_{i}_test.csv')

class DataRegisterer(object):
    def __init__(self):
        # Get credential for MLClient
        try:
            credential = DefaultAzureCredential()
        except Exception as ex:
            print("Please login first using the following command:")
            print("az login")
            raise ex

        # Read the config from the current directory and get 
        self.ws = MLClient.from_config(credential=credential)
    
    
    def register_table(self, data_dir, name, description):
        my_path = data_dir
        my_data = Data(
            path=my_path,
            type=AssetTypes.URI_FILE,
            description=description,
            name=name
        )
        
        self.ws.data.create_or_update(my_data)

    def register_folder(self, data_dir, name, description):
        my_path = data_dir
        my_data = Data(
            path=my_path,
            type=AssetTypes.URI_FOLDER,
            description=description,
            name=name
        )
        
        self.ws.data.create_or_update(my_data)


def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_