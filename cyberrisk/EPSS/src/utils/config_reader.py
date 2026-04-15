import configparser,json
from configparser import ConfigParser
import datetime

class ConfigParserEPSS(ConfigParser):
    def __init__(self, filepath=None):
        super().__init__()
        self.optionxform = str
        self.set_defaults()
        if filepath:
            self.read(filepath)
        self.model = self["MODEL"]["model"]
        self.modules = {k: self.process(v, k, "MODULES")
                        for k, v in self["MODULES"].items()}
        self.nvd_features = {k: self.process(v, k, "NVD FEATURES")
                             for k, v in self["NVD FEATURES"].items()}
        self.observation_features = {k: self.process(v, k, "OBSERVATION FEATURES")
                                      for k, v in self["OBSERVATION FEATURES"].items()}
        self.other_features = {k: self.process(v, k, "OTHER FEATURES")
                                for k, v in self["OTHER FEATURES"].items()}
        self.horizon = self.process(self["TARGET"]["horizon"])
        if self["DATA"]["NVD"]:
            self.nvd_filepath = self["DATA"]["NVD"]
        else:
            raise Exception("Please provide a filepath to NVD json data in config file.")
        if "observations" in self["DATA"].keys():
            self.observations_filepath = self["DATA"]["observations"]
        else:
            raise Exception("Please provide a filepath to json data with observed counts in config file.")
        if "NVD feature CSV" in self["DATA"].keys():
            self.nvd_feature_csv = self["DATA"]["NVD feature CSV"]
        else:
            self.nvd_feature_csv = False
        if "observations feature CSV" in self["DATA"].keys():
            self.observations_feature_csv = self["DATA"]["observations feature CSV"]
        else:
            self.observations_feature_csv = False

        if "EPSS prediction outpath" in self["DATA"].keys():
            self.predictions_outpath = self["DATA"]["EPSS prediction outpath"]
        else:
            self.predictions_outpath = False
        self.date = self.process(self["TARGET"]["date"])
        
    def set_defaults(self):
        self["MODEL"] = {"model": "ClassifierRF"}
        self["MODULES"] = {"ClassifierCWE": False,
                           "ConverterCVSS": False,
                           "ClassifierCVSS": False,
                           "CombinerCVSS": False}
        self["NVD FEATURES"] = {"age": True,
                                "cwe_id": True,
                                "cvss_elements": "[]"}
        self["OBSERVATION FEATURES"] = {"mean_counts": "[30]"}
        self["OTHER FEATURES"] = {"embed_description": False}

        self["DATA"] = {"NVD": "",
                        "observations": "",
                        "NVD feature CSV": ""}
        self["TARGET"] = {"horizon": "30"}
        self["TARGET"] = {"date": "2023/6/20"}

    def process(self, x: bool, y: str=None, s: str=None):
        if x == "true" or x == "false":
            return self[s].getboolean(y)
        elif x.startswith("["):
            return json.loads(self.get(s, y))
        elif len(x.split("/")) == 3:
            items = x.split("/")
            return datetime.datetime(int(items[0]), int(items[1]), int(items[2]))
        else:
            try:
                return int(x)
            except:
                return x
