from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

import pandas as pd
import numpy as np


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

    new_columns = set(df.columns)

    # output_columns = ["IID"]
    output_columns = list(new_columns - original_columns)
    df = df[output_columns]

    return df

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