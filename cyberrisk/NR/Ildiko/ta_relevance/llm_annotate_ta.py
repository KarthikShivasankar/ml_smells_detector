""" Script for (1) selecting a SCIO subset relevant for threat actor (TA) profiling based on search terms 
and for (2) annotating this subset automatically with an instruction fine-tuned large language model (LLM).
"""

import os
import transformers
import torch
import time
import json
import zipfile
import copy
from datetime import datetime
from collections import Counter
import spacy_fastlang
import spacy
from mitreattack.stix20 import MitreAttackData
import random
import argparse

# LLM Prompt for TA relevance annotation on a 3-point scale. One-shot setup, that is with one example per each score.
TA_PROMPT_1SHOT = [
    { "role": "user", "content": """Your task is to rate the relevance of a cybersecurity document for profiling a threat actor on a scale from 1 to 3, where:

3 (Very Relevant): The document contains comprehensive details for profiling a threat actor including its origin, motivation, usual targets and general attack patterns.
2 (Somewhat Relevant): The document provides some useful information about a single threat actor, but it is not comprehensive. The focus is on one or more of its previous attacks. 
1 (Not Relevant): The document mentions a threat actor, but the main focus is not on this, but on either event details, multiple threat actors, victims or risk mitigating measures. 

Please provide only a score as response, see the examples below with a document snippet and a score.

Doc: What are Advanced Persistent Threats?\nWhere are APTs located?\nRussia: Fancy Bear, GRU, FSB, Conti, and more\nChina: CISA advisories and ties to the Chinese People’s Liberation Army\nIran: MuddyWater and state-sponsored ransomware\nNorth Korea: Specialized training and the Guardians Of Peace\nTrack threat actor activity with Flashpoint\n\n\n\n\n\nWhat are Advanced Persistent Threats?\n\n\n\n\nAn Advanced Persistent Threat (APT) is a malicious actor who possesses extraordinary skill and resources—enabling them to infiltrate and exfiltrate an organizations’ network. APTs use a variety of techniques, tactics, and tools—such as highly-targeted social engineering attacks, ransomware, vulnerability exploits, and zero-days to accomplish their illicit objectives.
"""}, # 6ab5db414a7c3d19054db5a13c7e044059ce0c6285dbd1fc2500dbe086c94658
    { "role": "assistant", "content": "Score: 1"},
    { "role": "user", "content": "Doc: Winter Vivern (aka TA473), a Russian hacking group, has been exploiting vulnerabilities (CVE-2022-27926) in unpatched Zimbra instances to access the emails of NATO officials, governments, military people, and diplomats. The CVE-2022-27926 flaw affects versions 9.0.0 of Zimbra Collaboration, which is used to host webmail portals with public access. "}, # d816ba6cbbde220e849c8766fd629ee8413f777575581c6316b8d7af21a10df0
    { "role": "assistant", "content": "Score: 2"},
    { "role": "user", "content": "Doc: A new North Korean cyber operator has been attributed to a series of attacks conducted to gather strategic intelligence aligned with the state’s geopolitical interests. Security researchers, which are tracking the threat group’s activity under the moniker APT43, believe that the group’s main purpose is espionage, but its attacks are also financially-motivated."}, # 128b2130659e59cf3ed43b8401b5e197fe39794efb8e062dcac32d7dae126b81
    { "role": "assistant", "content": "Score: 3"},
    { "role": "user", "content": "Doc: {}"} # Each instance to annotate will be plugged in here
]

# Ids of the SCIO docs used as examples in the prompt
EX_TEXT_IDS = ['6ab5db414a7c3d19054db5a13c7e044059ce0c6285dbd1fc2500dbe086c94658',
               'd816ba6cbbde220e849c8766fd629ee8413f777575581c6316b8d7af21a10df0',
               '128b2130659e59cf3ed43b8401b5e197fe39794efb8e062dcac32d7dae126b81']

# Generic terms related to threat actors
TA_TERMS = ["APT ", "Advanced Persistent Threat", "hacking group", "hacker group", "threat group", "threat actor", 
"cyber espionage group", "cybercriminal group", "malicious actor",  "malicious group", "attack group", "intrusion set"]

# Configurations for quantization
bnb_config_4bit = transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                  #bnb_4bit_use_double_quant=True, # nested quantization for more memory efficiency (quantization constants from the first quantization are quantized again)
                                                  bnb_4bit_quant_type="nf4",      # quantization data type in the bnb.nn.Linear4Bit layers
                                                  bnb_4bit_compute_dtype=torch.bfloat16 # default torch.float32 reduced for speedup
                                                  )

def load_tokenizer_and_model(llm_name, bnb_config={}):
    """ Load a pre-trained tokenizer and model and return them both.
    Parameters:
        llm_name (str): Huggingface name (ID) of model, e.g. 'google/gemma-2b-it'
        bnb_config (dict): bitsandbytes config for quantization
    Returns:
        tuple: the tokenizer and model objects
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(llm_name,  quantization_config=bnb_config, device_map="auto")
    if bnb_config:    
        return tokenizer, model
    else:
        return tokenizer, model.to('cuda') # needed only for non-quantized models

def inference(chat, tokenizer, model):
    """
    Use a pre-trained tokenizer and model to perform inference using the provided chat input 
    with a template applied to it matching the model.
    Parameters:
        chat (list): The input chat for which a response is to be generated. Should be a list
                     of dictionaries with 'role' ('user' or 'assistant') and 'content' (the message) keys.
        tokenizer (object): A pre-trained tokenizer object capable of tokenizing input chats.
        model (object): A pre-trained model capable of generating responses based on input chats.
    Returns:
        str: The generated response for the input chat.
    """
    start = time.time()
    input_ids =  tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda") 
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]  # llama3-specific
    outputs = model.generate(input_ids, 
                             max_new_tokens=10, # max length of the output to generate
                             eos_token_id=terminators,
                             #pad_token_id=tokenizer.eos_token_id,
                             do_sample=True,    # sample based generation for more varied output
                             temperature=0.3,   # allow for some but not too much creativity in output 
                             use_cache=True) 
                                 
    # Extract the generated tokens
    input_length = input_ids.shape[1]

    # Retain only the generated tokens and omit the input repeated in the output
    generated_tokens = outputs[:, input_length:]  
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print("Inference time =", round(time.time() - start, 2))
    print()

    return response

def load_ta_groups(path_to_list):
    "Load threat group list from https://attack.mitre.org/groups/."
    with open(path_to_list) as f:
        ta_groups = [t.strip() for t in f.readlines()]
    return list(set(ta_groups))

def get_mitre_ta_groups(mitre_json_path="enterprise-attack.json"):
    """
    Documentation: https://github.com/mitre-attack/mitreattack-python/blob/master/docs/mitre_attack_data/mitre_attack_data.rst
    mitre_json_path: JSON downloaded from https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json 
    """
    ta_names = [] 
    ta_aliases = {}
    mitre_attack_data = MitreAttackData(mitre_json_path)
    groups = mitre_attack_data.get_groups()
    for group in groups:
        # Collect each group's name and any aliases
        if 'name' in group:
            name = group['name']
            if name not in ta_names:
                ta_names.append(name)
                if 'aliases' in group:
                    ta_aliases[group['name']] = list(set(group["aliases"])) 
    # TODO: use descriptions for something? (+ check if type is attack type?)
    return ta_names, ta_aliases

def denoise(text, min_len=100, threshold=0.5):
    "Filter text for paragraphs with a high proportion of non-alphabetical characters."
    cleaned_text = ""
    blocks = text.split("\n")
    for block in blocks:
        if len(block) > min_len:
            non_alpha = [char for char in block if not char.isalpha()]
            if len(non_alpha) / len(block) < threshold:
                cleaned_text += block.strip() + "\n"
    return cleaned_text

def load_manual_set():
    "Load manually labeled data to exclude from LLM annotation."
    manual_set = {}
    with open("ta_manual_pilot.txt") as f:
        lines = f.readlines()
        for line in lines:
            fname, score = line.strip().split(" ")
            manual_set[fname] = int(score)
    return manual_set

def get_doc_lg(text, nlp, prob_threshold=0.8):
    """Detect document language with Spacy.
    Parameters:
        text: str; input text to analyze
        nlp: obj; loaded Spacy processing pipeline
        prob_threshold: float; minimum value of probability (confidence) for language classification
    Returns:
        str; the Spacy-predicted language of the document
    """
    proc_doc = nlp(text)
    if proc_doc._.language_score >= prob_threshold:
        return proc_doc._.language

def get_main_TAs(terms, ta_aliases):
    """ Get TAs frequently mentioned based on cumulative count of mentions.
    Parameters:
        terms: list; any TA related term found in a doc
        ta_aliases: list; list of TA aliases
    Returns:
        list of TA names with more than 2 mentions (including alias metions)
    """
    main_tas = []
    for term in terms:
        if term not in TA_TERMS:
            cumul_cnt = 0
            if term in ta_aliases:
                cumul_cnt +=  terms[term]
                for alias in ta_aliases[term]:
                    if alias in terms and alias != term:
                        cumul_cnt +=  terms[alias]
            if cumul_cnt > 2:
                main_tas.append(term)
    return main_tas

def get_scio_TA_subset(scraped_dir_scio, start_date, end_date, search_terms, ta_aliases, file_limit, outfile):
    """ Loads and selects a subset of the scraped SCIO data. 
    It does some filtering and runs a keyword search to
    find threat actor relevant texts. Save a subset of the data
    with TA relevant text plus twice the amount of less relevant texts.
    Information about the detected terms and their frequency is also added.
    Parameters:
        scraped_dir_scio: str; path to the scraped (zipped) SCIO dump 
        start_date: str; start doc date in '%Y-%m-%d%z' strptime format, e.g. "1990-01-01+02:00"
        end_date: str; end doc date in '%Y-%m-%d%z' strptime format, e.g. "1990-01-01+02:00"
        search_terms: list of TA-relevant terms to search for in doc 
        ta_aliases: dict; a mapping between a TA name and all their alternative names (aliases)
        file_limit: int; maximum number of files to process
        outfile: str; path to JSON file where to save output
    Returns:
        Saves a JSON with the SCIO subset.
    """
    print('Loading data...')

    # Initialize NLP processing pipeline (used for language detection)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector") # based on spacy_fastlang

    # Convert dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d%z') 
    end_date = datetime.strptime(end_date, '%Y-%m-%d%z')
    
    texts = []
    added_data = [] # used for filtering duplicates based on title and text
    dates = [] # dev only
    less_relevant_texts = []
    manual_data = load_manual_set()
    manual_data_all_info = []

    # Add TA aliases to search terms
    for k,v in ta_aliases.items():
        search_terms.extend(v)
    print(len(search_terms))

    # Process and check each file
    with zipfile.ZipFile(scraped_dir_scio, 'r') as zip:
        zip_files = zip.namelist()
        print('Nr zip files', len(zip_files))
        if file_limit:
            zip_files = zip_files[:file_limit]
        for zip_file in zip_files:
            filename, extension = os.path.splitext(zip_file)
            
            # Exclude example texts in prompt
            if filename not in EX_TEXT_IDS:
                print(f"Processing {filename} ...")
                
                # Read file without extracting it
                json_str = zip.read(zip_file)
                json_obj = json.loads(json_str)
                date = json_obj['Creation-Date']
                if type(date) == list:
                    date = sorted(date)[0]
                try: 
                    doc_date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z')
                except ValueError:
                    doc_date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')

                # Filter by date
                if doc_date >= start_date and doc_date <= end_date:
                    text = json_obj['content']
                    text = denoise(text)

                    # Get title if any
                    if 'title' in json_obj['metadata']: 
                        title = json_obj['metadata']['title']
                        # If multiple titles, take only first to avoid data loading issues later with mixed string and list vals 
                        if type(title) == list:
                            title = title[0]
                        else:
                            title = title
                    else:
                        title = ''

                    # Deduplicate
                    if text not in added_data and title not in added_data:
                        doc_obj = {}
                        doc_obj['text'] = text

                        # Filter for length
                        if len(doc_obj["text"]) > 100 and len(doc_obj["text"]) < 10000:
                            doc_lg = get_doc_lg(doc_obj['text'], nlp)

                            # Filter for language (English and Norwegian)
                            if doc_lg in ['en', 'no', 'nb']:
                                doc_obj['lang'] = doc_lg
                                doc_obj['date'] = str(doc_date.date())
                                doc_obj['filename'] = filename
                                doc_obj['title'] = title

                                # Store manually annotated texts separately w their score
                                if filename in manual_data:
                                    doc_obj["manual_score"] = manual_data[filename]
                                    manual_data_all_info.append(doc_obj)
                                else:

                                    # Scan texts for search terms 
                                    term_count = []
                                    terms = {}
                                    for term in search_terms:
                                        if term in text:
                                            cnt = text.count(term)
                                            terms[term] = cnt
                                            term_count.append(cnt)
                                    if terms:
                                        print(terms)
                                    doc_obj['terms'] = terms #IDEA: correlate later llm score to the lens of this
                                
                                    # TODO: check if needed and bug fix
                                    # main_TAs = get_main_TAs(terms, ta_aliases)
                                    # if main_tas:
                                    #     print("\t", main_tas)
                                    # doc_obj['main_TAs'] = main_tas
                                    
                                    # Separate more and less relevant docs
                                    term_count = sum(term_count) 
                                    if term_count > 3: 
                                        texts.append(doc_obj)      
                                        print("REL", term_count)                  
                                    else:
                                        less_relevant_texts.append(doc_obj) 
                                    if title:
                                        added_data.append(title)
                                    added_data.append(text)
                                    dates.append(doc_date.date())

    print(sorted(dates)[0], sorted(dates)[-1])
    print(f"Found {len(texts)} relevant texts.")

    # Merge relevant and lessrelevant docs (w fewer term matches)
    nr_less_rel = 2*len(texts)
    random.Random(123).shuffle(less_relevant_texts)
    if len(less_relevant_texts) > nr_less_rel:
        subset = texts + less_relevant_texts[:nr_less_rel] 
    else:
        subset = texts + less_relevant_texts

    with open(outfile, 'w', encoding='utf-8') as f:
      json.dump(subset, f, ensure_ascii=False, indent=4)
    
    # Save manually labeled set (done only once if missing)
    if 'scio_ta_manual_set.json' not in os.listdir():
        with open('scio_ta_manual_set.json', 'w', encoding='utf-8') as f:
            json.dump(manual_data_all_info, f, ensure_ascii=False, indent=4)

def get_llm_annotations(data_path, tokenizer, model, outfile):
    """Load documents and pass each to an LLM for annotation. 
    Saves a json with the original data updated with the predicted LLM score.
    Parameters:
        data_path: str; path to the selected SCIO subset in a JSON format  
        tokenizer: loaded tokenizer 
        model: loaded LLM 
        outfile: str; path to JSON file where to save the LLM-annotated subset
    Returns:
        Saves the LLM-annotated subset to a JSON file.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        ann_dataset = []
        for doc_obj in dataset:
            print("IN:", doc_obj["text"][:80])

            # Prepare prompt
            prompt = copy.deepcopy(TA_PROMPT_1SHOT)
            prompt[-1]['content'] = prompt[-1]['content'].format(doc_obj["text"])
            output = inference(prompt, tokenizer, model)

            # Extract score from LLM output
            output = output.replace("assistant","").lstrip()
            
            if output.startswith("Score:"):
                score = output.replace("Score:", "").lstrip()
                # Try to convert score to an integer and if possible, save
                try:
                    score = int(score)
                    doc_obj["llm_score"] = score
                    ann_dataset.append(doc_obj)
                    print("OUT:", doc_obj["llm_score"])
                except:
                    pass
            else:
                print("OUT wf:", output)

            # Save JSON updated with LLM-predicted TA relevance score 
            with open(outfile, 'w', encoding='utf-8') as f:
                json.dump(ann_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--data_path", type=str, 
                           help="Path to TAB dataset.", required=True)
    argparser.add_argument("-a", "--action", type=str, 
                            help="Choose 'sel' for only selecting a doc subset, 'ann' for annotating, and 'all' for both (default).", 
                            default="all")
    argparser.add_argument("-llm", "--llm_name", type=str, 
                            help="Huggingface name for the LLM to use.", 
                            default="meta-llama/Meta-Llama-3-8B-Instruct")
    argparser.add_argument("-max", "--max_doc", type=int, 
                            help="Maximum number of documents to process.", 
                            default=None)
    argparser.add_argument("-sd", "--start_date", type=str, 
                            help="Start date of documents to include in subset.", 
                            default="1990-01-01+02:00")
    argparser.add_argument("-ed", "--end_date", type=str, 
                            help="End date of documents to include in subset.", 
                            default="2023-12-31+02:00")
    argparser.add_argument("-os", "--out_subset", type=str, 
                            help="Path to JSON where to save the selected doc subset.", 
                            default="scio_ta_ds_TEST.json")
    argparser.add_argument("-oa", "--out_llm_annot", type=str, 
                            help="Path to JSON where to save the annotated docs.", 
                            default="scio_ta_ds_llm_ann_TEST.json")
    args = argparser.parse_args()

    ## Load threat actor names and merge term lists
    ta_groups = load_ta_groups("mitre_threat_groups.txt")
    mitre_ta_groups, mitre_ta_aliases = get_mitre_ta_groups()
    for ta in ta_groups:
        if ta not in mitre_ta_groups:
            mitre_ta_groups.append(ta)
    search_terms = mitre_ta_groups + TA_TERMS
    
    if args.action not in ['sel', 'ann', 'all']:
        raise ValueError("Parameter -a(ction) should be 'sel', 'ann' or 'all'.")
    else:
        if args.action != 'ann':
            ## Create data subset to annotate
            texts = get_scio_TA_subset(args.data_path, args.start_date, args.end_date, search_terms, mitre_ta_aliases, args.max_doc, args.out_subset)
        if args.action != 'sel':
            ## LLM annotation
            tokenizer, model = load_tokenizer_and_model(args.llm_name, bnb_config_4bit)
            get_llm_annotations(args.out_subset, tokenizer, model, args.out_llm_annot) # annotate the unannotated (& unfiltered) SCIO subset