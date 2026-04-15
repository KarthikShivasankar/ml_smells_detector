from .config_reader import ConfigParserEPSS

def dynamic_import(module_name: str, class_name: str) -> any:
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError:
             exit("Module (" + module_name + ") not found.")
    try:
        clss = getattr(module, class_name)
    except ImportError:
        exit("Class (" + class_name + ") not found in module (" + module_name + ").")
    return clss


def import_dependenices(module: str) -> None:
    module2dependencies = {"ClassifierCWE": ["SentenceTransformer"],
                           "ClassifierCVSS": ["SentenceTransformer"]}
    

def display_settings(mode: str, config: ConfigParserEPSS) -> None:
    print("EPSS")
    print("------------------------------------------------------------------------------------")
    print("MODEL TYPE:\t\t".expandtabs(6) + config.model + " EPSS model")
    print("MODE: \t\t".expandtabs(6) + mode)
    modules = [m for m, v in config.modules.items() if v]
    if modules:
        print("MODULES:\n\t".expandtabs(6) + ", ".join(modules))
    else:
        print("MODULES:\n\t".expandtabs(6) + "None")
    print("TARGET:\n\t".expandtabs(6) + "Horizon:\t".expandtabs(6) + str(config.horizon) + "\n\t".expandtabs(6) + "Target date:\t".expandtabs(6) + str(config.date))

    
    nvd_features = [str(f)+": "+str(v) for f, v in config.nvd_features.items() if v]
    nvd_features = [f.replace(": True", "") for f in nvd_features]
    nvd_features = [f.replace(": False", "") for f in nvd_features]
    print("NVD FEATURES:\n\t".expandtabs(6) + ", ".join(nvd_features))

    observation_features = [str(f)+": "+str(v) for f, v in config.observation_features.items() if v]
    observation_features = [f.replace(": True", "") for f in observation_features]
    observation_features = [f.replace(": False", "") for f in observation_features]
    print("OBSERVATION FEATURES:\n\t".expandtabs(6) + ", ".join(observation_features))

    other_features = [str(f)+": "+str(v) for f, v in config.other_features.items() if v]
    other_features = [f.replace(": True", "") for f in other_features]
    other_features = [f.replace(": False", "") for f in other_features]
    print("OTHER FEATURES:\n\t".expandtabs(6) + ", ".join(other_features))
    
    print("DATA:\n\t".expandtabs(6) + "NVD data:\t\t'".expandtabs(6) + config.nvd_filepath + "'\n\t".expandtabs(6) + "Observed counts:\t'".expandtabs(6) + config.observations_filepath+ "'\n\t".expandtabs(6) + "NVD features cvs:\t'".expandtabs(6) + config.nvd_feature_csv+ "'\n\t".expandtabs(6) + "Observation features csv:\t'".expandtabs(6) + config.observations_feature_csv+"'")
    print("------------------------------------------------------------------------------------\n")

