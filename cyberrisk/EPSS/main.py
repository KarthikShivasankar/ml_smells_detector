from argparse import ArgumentParser
from src.utils import ConfigParserEPSS, dynamic_import, display_settings, get_features
import numpy as np
import random
from collections import Counter
import pandas as pd


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("mode", choices=["train", "eval", "predict"])
    argparser.add_argument("--config", required=True)
    args = argparser.parse_args()
    config = ConfigParserEPSS(args.config)
    ModelClass = dynamic_import("src.models", config.model)
    model = ModelClass(config)
    if model.lm:
        config.nvd_features["description"] = True
    display_settings(args.mode, config)
    print("GETTING INPUT DATA")
    features = get_features(config)
    print("---------------------")
    if args.mode == "train":
        print("TRAINING EPSS MODEL")
        model.train(features)
        model.save("test.model")
    elif args.mode == "eval":
        print("EVALUATING EPSS MODEL")
        model.load("test.model")
        model.evaluate(features)
    elif args.mode == "predict":
        print("PREDICTING EPSS SCORES")
        model.load("test.model")
        EPSS_scores = model.predict_epss(features)
        output_df = pd.DataFrame()
        CVEs = features["cve"]
        output_df["cve"] = CVEs
        output_df["epss"] = EPSS_scores
        print("Saving scores to:\n\t" + config.predictions_outpath)
        output_df.to_csv(config.predictions_outpath)
        
    
