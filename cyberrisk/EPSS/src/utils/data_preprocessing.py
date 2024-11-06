import pandas as pd
import numpy as np
import os, numbers
from .nvd_data_preprocessing import add_missing_nvd_features, read_NVD_directory, get_missing_nvd_features, get_nvd_feature_column_names, save_dataframe
from .observation_data_preprocessing import read_observations, process_observation_data, get_observation_feature_column_names, get_missing_observation_features, add_missing_observation_features
from .config_reader import ConfigParserEPSS

def get_features(config: ConfigParserEPSS) -> pd.DataFrame:
    nvd_features = get_nvd_features(config)
    observation_features = get_observation_features(config)
    features = pd.merge(nvd_features, observation_features, on="cve", how="left")
    for col in observation_features.columns:
            features[col] = features[col].fillna(0)
    return features

def get_nvd_features(config: ConfigParserEPSS) -> pd.DataFrame:
    if config.nvd_feature_csv and os.path.exists(config.nvd_feature_csv):
        print("LOADING PRE-EXISTING NVD FEATURE DATAFRAME")
        feature_dataframe = pd.read_csv(config.nvd_feature_csv)
        missing = get_missing_nvd_features(feature_dataframe, config)
        if missing:
            print("missing features:", [m for m, v in missing.items()])
            feature_dataframe = add_missing_nvd_features(feature_dataframe, missing, config)
            save_dataframe(feature_dataframe, config.nvd_feature_csv)
    else:
        print("GENERATING NEW NVD FEATURE DATAFRAME")
        feature_dataframe = read_NVD_directory(config)
        if config.nvd_feature_csv:
            save_dataframe(feature_dataframe, config.nvd_feature_csv)
    return feature_dataframe

def get_feature2id_dictionary(data: pd.DataFrame, feature: pd.DataFrame) -> dict:
    unique_features = list(set(data[feature]))
    if isinstance(unique_features[0], numbers.Number):
        return None
    unique_features.sort()
    return {f: i for i, f in enumerate(unique_features)}

def numericalise_features(data: pd.DataFrame, config: ConfigParserEPSS) -> np.array:
    columns = get_nvd_feature_column_names(config.nvd_features)
    columns += get_observation_feature_column_names(config.observation_features)
    numericalised_features = np.zeros((len(data), len(columns)))
    column2dict = {c: get_feature2id_dictionary(data, c)
                   for c in columns}
    for i, c in enumerate(columns):
        if column2dict[c] is None:
            numericalised = data[c]
        else:
            numericalised = [column2dict[c][entry]
                             for entry in data[c]]
        numericalised_features[:,i] = numericalised
    return numericalised_features


def get_observation_features(config: ConfigParserEPSS) -> pd.DataFrame:
    if config.observations_feature_csv and os.path.exists(config.observations_feature_csv):
        print("LOADING PRE-EXISTING OBSERVATIONS FEATURE DATAFRAME")
        observation_features_dataframe = pd.read_csv(config.observations_feature_csv)
        missing = get_missing_observation_features(observation_features_dataframe, config)
        if missing:
            print("missing features:", [m for m, v in missing.items()])
            observation_features_dataframe = add_missing_observation_features(observation_features_dataframe, missing, config)
            save_dataframe(observation_features_dataframe,
                           config.observations_feature_csv)
    else:
        print("GENERATING NEW OBSERVATION FEATURE DATAFRAME")
        observation_features_dataframe = process_observation_data(config)
        if config.observations_feature_csv:
            save_dataframe(observation_features_dataframe,
                           config.observations_feature_csv)
    return observation_features_dataframe
