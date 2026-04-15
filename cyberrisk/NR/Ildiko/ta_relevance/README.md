## General

Code for finding automatically relevant documents for Threat Actor Profiling. Includes code for dataset preparation, model traning and automatic annotation with a large language model (LLM). 

## Files

- llm_annotate_ta.py: selects a subset of relevant SCIO documents and automatically annotates them with an LLM.
- doc_relevance.py: training of an ordinal regression model based on BERT for predicting a TA relevance score  
- enterprise-attack.json: list of TA groups in a JSON downloaded from https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json 
- mitre_threat_groups.txt: threat group list from https://attack.mitre.org/groups/
- ta_manual_pilot.txt: manually annotate pilot data (one doc_id and score per line, separated by space)

## Dependencies

Main libraries and versions used:
transformers 4.44.0
spacy 3.7.6
spacy_fastlang 2.1.0
pandas 2.2.2
scikit-learn 1.5.1
mitreattack-python 3.0.6

(You have to run `python -m spacy download en_core_web_sm` for downloading the spacy model required if you don't already have it.) 

See requirements.txt for the complete list of packages from the virtual environment used.

## Example runs

- Select a subset of SCIO documents and LLM-annotate them for threat actor relevance
`python llm_annotate_ta.py -a all -d sciodump.zip -max 50`

- Train a BERT model for threat actor relevance and evaluate it on the test set
`python doc_relevance.py -m MODEL_OUT_DIR -a tep`

- Predict data (test set) and plot a confusion matrix 
`python doc_relevance.py -m  MODEL_OUT_DIR/checkpoint-XXX/model.safetensors -a p`
