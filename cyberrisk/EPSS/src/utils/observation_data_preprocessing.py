import pandas as pd
import datetime
import numpy as np
from .config_reader import ConfigParserEPSS

def add_formatted_dates(observations: pd.DataFrame) -> pd.DataFrame:
    formatted_dates = []
    for i, row in observations.iterrows():
        date_items = [int(c) for c in row["date"].split("-")]
        date = datetime.datetime(date_items[0],
                                 date_items[1],
                                 date_items[2])
        formatted_dates.append(date)
    observations["formatted_date"] = formatted_dates
    return observations

def read_observations(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = add_formatted_dates(df)
    return df

def get_missing_observation_features(df: pd.DataFrame, config: ConfigParserEPSS) -> dict|bool:
    print("Checking all selected features are present in observations feature dataframe.")
    columns = set(df.columns)
    features = set(get_observation_feature_column_names(config.observation_features))
    missing = list(features.difference(columns))
    if missing:
        return convert_missing_column_back_to_formated_feature(missing, config)
    else:
        return False

def convert_missing_column_back_to_formated_feature(missing: list, config: ConfigParserEPSS) -> dict:
    ### need to reformat missing values into expect format
    formatted_missing = {}
    for m in missing:
        for k, v in config.observation_features.items():
            if k in m:
                if isinstance(v, list):
                    data_type = type(v[0])
                    if k not in formatted_missing.keys():
                        formatted_missing[k] = [data_type(m.split("_")[-1])]
                    else:
                        formatted_missing[k].append(data_type(m.split("_")[-1]))
                else:
                    formatted_missing[m] = True
    return formatted_missing

def get_observation_feature_column_names(features: dict) -> list:
    columns = []
    for feature, val in features.items():
        if not val:
            continue
        if isinstance(val, list):
            for v in val:
                columns.append(feature+"_"+str(v))
        else:
            columns.append(feature)
    return columns

def add_missing_observation_features(df: pd.DataFrame, features: dict, config: ConfigParserEPSS) -> pd.DataFrame:
    cve2features = {}
    columns = get_observation_feature_column_names(features)
    column2list = {c: [] for c in columns}
    print(config.observations_filepath)
    observations = read_observations(config.observations_filepath)
    CVEs = list(set(observations["cve"]))
    CVEs.sort()
    column2function = get_column2function(features)
    column2list = {c: [] for c in column2function}
    df2 = pd.DataFrame()
    for cve in CVEs:
        for col, (func, val) in column2function.items():
            column2list[col].append(func(observations, cve, val, config.date))
    df2["cve"] = CVEs
    for c, l in column2list.items():
        df2[c] = l
    df = pd.merge(df, df2, on="cve", how="left")
    return df

def process_observation_data(config: ConfigParserEPSS) -> pd.DataFrame:
    df = pd.DataFrame()
    date = config.date
    observations = read_observations(config.observations_filepath)
    observations = add_formatted_dates(observations)
    column2function = get_column2function(config.observation_features)
    column2list = {c: [] for c in column2function}
    CVEs = list(set(observations["cve"]))
    CVEs.sort()
    for cve in CVEs:
        for col, (func, val) in column2function.items():
            column2list[col].append(func(observations, cve, val, date))
    df["cve"] = CVEs
    for c, l in column2list.items():
        df[c] = l
    return df
    
def get_column2function(features: dict) -> dict:
    c2f = {}
    for feature, val in features.items():
        print(feature, val)
        if not val:
            continue
        if isinstance(val, list):
            for v in val:
                c2f[feature+"_"+str(v)] = (FEATURE2FUNCTION[feature],
                                             v)
    return c2f

def get_mean_count(observations: pd.DataFrame, cve: str, period: int, date: datetime.datetime) -> float:
    df = observations[observations["cve"]==cve]
    counts = np.zeros(period)
    i = 0
    for j, row in  df.iterrows():
        cve_day = row["formatted_date"]
        if cve_day >= date:
            continue
        if (date - cve_day).days <= period:
            counts[i] = row["count"]
            i += 1
    return np.mean(counts)




FEATURE2FUNCTION = {"mean_counts":  get_mean_count,
                    }
