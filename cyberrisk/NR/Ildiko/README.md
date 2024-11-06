## General

Code for the detection of threat event scenario components with two approaches: 
- a simpler, keyword-based approach relying on [KeyBERT](https://maartengr.github.io/KeyBERT/index.html) and 
- a Named Entity Recognition (NER) and relation extraction (REL) based approach that produces categorized information relevant to threat scenarios (trained on the [CASIE](https://github.com/Ebiquity/CASIE) corpus.

## Modules

- get_keywords.py: extracts keywords including some post-processing (e.g. removing similar ones or duplicates), saves them  in a CSV file and counts their occurrence in the source texts.
- process_casie.py: extracts entity and relation annotations from CASIE and creates an dataset usable to train Spacy-based NER and REL models
- test_detection.py: apply a NER model trained with CASIE annotations on any text
- extract_threat_info.py: a demo for applying both NER and REL on a document
- rel_model.py: for relation extraction, borrowed from [ExplosionAI tutorial](https://github.com/explosion/projects/tree/v3/tutorials/rel_component) 
- rel_pipe.py: for relation extraction, borrowed from [ExplosionAI tutorial](https://github.com/explosion/projects/tree/v3/tutorials/rel_component)

## Dependencies

- keybert 0.7.0
- spacy 3.5.3
- spacy_fastlang 1.0.1
- Levenshtein 0.20.9
- pandas 1.5.3
- Python 3.10.9

(You have to run `python -m spacy download en_core_web_sm` for downloading the spacy model required if you don't already have it.) 

These packages should be enough for the modules included here, but there is also a pip-generated requirements.txt file for replicating the exact environment used (contains some other packages too used for additional tests not included here).

## Keyword detection

- Example call for running keyword extraction:

`python get_keywords.py -data_dir sciodump.zip -start_date 2023-03-01 -end_date 2023-04-01`

- Get statistics about docs including language detection, without extracting keywords:

`python get_keywords.py -data_dir sciodump.zip -start_date 2023-03-01 -end_date 2023-04-01 -skip_extraction -detect_lg`

## NER-based threat scenario component detection

As entity labels we used both argument type labels and event type labels (the CASIE paper used separate classifiers for these).

### Dataset preparation

Create a a Spacy-compatible docbin dataset (a train, development and test split) with entity and relation labels from a downloaded version of the CASIE corpus:

`python process_casie.py -data_dir CASIE-master/CASIE-master/data/ -task all -with_relations`

Alternatively, the already prepared dataset files under the 'casie_all_w_relations_mapped' (or 'casie_ner' for NER only) directory can also be used for training and testing.

### Training with Spacy (GPU)

Train a NER model on the CASIE entity annotations

`python -m spacy train -g 0 casie_ner/ner_config.cfg --output ./casie_ner/output`

The trained model can also be made available upon request outside GitHub. It has a performace of F1=63.24 (some CASIE annotations were excluded from the data used for training due to overlapping span offsets incompatible with Spacy's NER. The dataset used had a total of 831 documents across the 3 splits.)

The relation extraction training was done following the [ExplosionAI tutorial](https://github.com/explosion/projects/tree/v3/tutorials/rel_component)  

### NER Inference (GPU)

Evaluate NER with Spacy on the CASIE test set (with the trained model assumed to be under casie_ner/output) and generate some example html visualizations with color-highlighting:

`python -m spacy benchmark accuracy casie_ner/output/model-best casie_ner/test.spacy --output casie_ner/test_meta.json --gpu-id 0 --displacy-path html_vis --displacy-limit 10`

Do NER inference on a plain text file (with SCIO texts):

`python test_detection.py -model_dir casie_ner/output/model-best -test_data scio_test_data.txt -from_file`

Do NER inference on a short text directly from the command line with html visualizations using color-highlighting:

`python test_detection.py -model_dir casie_ner/output/model-best -test_data "QuoIntelligence followed the massive ransomware attacks targeting VMware ESXi servers worldwide by exploiting the two-year-old vulnerability CVE-2021-21974. While such servers should not be publicly exposed and should be patched by now, we observed how the attackers continue to exploit old vulnerabilities resulting in successful attacks."`

### NER+REL Inference (GPU)

Do NER+REL inference on a short text directly from the command line 

`python extract_threat_info.py -ner_path model-best_ner -rel_path model-best_rel -test_data "On April 14, the company disclosed to the California attorney general that a December 2015 breach compromised more sensitive information than first thought. It also disclosed new attacks from earlier this year that exposed names, contact information, email addresses and purchase histories, although  the retailer says it repelled most of the attacks. The dual notifications mark the latest problems for the company, which disclosed in early 2014 that its payment systems were infected with malware that stole 350,000 payment card details."`
