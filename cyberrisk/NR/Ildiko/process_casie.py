import os, codecs
import random
import json
import spacy
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from spacy.util import filter_spans
from collections import Counter
import argparse
import itertools

RELATION_LABELS = ['Attack-Pattern', 'Attacker', 'CVE', 'Capabilities', 'Compromised-Data', 
             'Damage-Amount', 'Discoverer', 'Issues-Addressed', 'Number-of-Data', 
             'Number-of-Victim', 'Patch', 'Patch-Number', 'Payment-Method', 'Place', 
             'Price', 'Purpose', 'Releaser', 'Supported_Platform', 'Time', 
             'Tool', 'Trusted-Entity', 'Victim', 'Vulnerability', 'Vulnerable_System', 
             'Vulnerable_System_Owner', 'Vulnerable_System_Version'] + ['Same'] # relation type among events only

SYMMETRIC_RELATIONS =  ['Same'] 

REL_LABEL_MAPPING = { 'Attacker': 'Attacker', 
                      'Victim' : 'Victim', 
                      'Vulnerable_System_Owner' : 'Victim',
                      'Releaser': 'Releaser', 
                      'Vulnerable_System': 'Asset',
                      'Compromised-Data': 'Asset', 
                      'Tool': 'Method', 
                      'Attack-Pattern': 'Method', 
                      'Capabilities': 'Method', 
                      'Purpose': 'Purpose',
                      'Same': 'Same' }

def convert_to_unspecified_rel(labels, mapping):
    extended_mapping = mapping
    for label in labels:
        if label not in mapping:
            extended_mapping[label] = 'Related_to'
    unique_mapped = list(set([v for k, v in extended_mapping.items()]))
    print('# unique mapped REL labels:', len(unique_mapped))
    return extended_mapping

def extract_text(fname):
    """ Extract texts from CASIE XML-like format with only texts 
    under the 'source/' directory.
    """
    with codecs.open(fname, 'r', 'utf8') as f:
        content = f.read()
        text_start = content.index('<text>')+6
        return content[text_start:-7] # drop closing </text> tag

def load_annotations(fname):
    " Load CASIE JSON data from the 'annotations/' directory. "
    with open(fname, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_casie(ann_dir, texts_dir, task="all", with_relations=False):
    """ Collect entity information (span start, end and label) from the 
    CASIE dataset (https://github.com/Ebiquity/CASIE). Both event nuggets
    and arguments are considered entities and their 'type' attribute is used
    as entity label. Optionally extract also relation labels for these entities.
    Returns a list with tuples of text and the list of entity information tuples.
    @ ann_dir: str; path to the CASIE JSON data with annotations;
    @ texts_dir: str; path to the CASIE text data in 'source/'.
    @ task: 'arg' for argument type classification,
            'event' for event type and realis classification.
    @ with_relations: bool; whether to include relation annotations 
    """
    data = [] 
    for fname in os.listdir(ann_dir):
        if fname != '10002.json':
            text_info = {}
            ann = load_annotations(os.path.join(ann_dir,fname))
            text = ann['content'] 
            text_txt = extract_text(os.path.join(texts_dir, fname.replace('json', 'txt')))

            # Checking length mismatch between .txt and .json texts
            # if len(text) != len(text_txt):
            #     if len(text_txt)-len(text) not in [3,4]:
            #         print(fname, len(text), len(text_txt), len(text_txt)-len(text))

            entities = [] 
            relations = []
            all_events_char_start = []
            for event_cluster in ann['cyberevent']['hopper']: # list of dict with related events ('Same' relation)  
                cluster_relation = event_cluster.get('relation', '')
                same_events_char_start = []
                for event in event_cluster['events']:

                    # Add event labels: type, (relalis), relation
                    event_type = event['type'] + event['subtype']
                    if event_type.endswith('Vulnerability'): 
                        event_type = event_type.split('-')[-1].replace('related', '')
                    event_realis = event['realis']
                    
                    # Merge 'Generic' and 'Other' classes
                    if event_realis == 'Generic':
                        event_realis = 'Other' 
                    
                    # Use joint arg type and binary realis labels
                    #event_type = event_realis + '_' + event_type
                    event_text = event['nugget']['text']
                    event_start = event['nugget']['startOffset']
                    event_end = event['nugget']['endOffset']
                    if task in ['event', 'all']:
                        entities.append((event_start, event_end, event_type))
                    elif task == 'realis':
                        entities.append((event_start, event_end, event_realis))
                    same_events_char_start.append(event_start)

                    # Add argument labels: (entity) type, relation
                    if task in ['arg', 'all']:
                        for arg in event.get('argument', []):
                            ent_label = arg['type']
                            if ent_label == 'PII':
                                ent_label = 'Data'
                            # if ent_label == 'Money':
                            #     print(ent_label, arg['text'])
                            entities.append((arg['startOffset'], arg['endOffset'], ent_label))
                            if with_relations:
                                relation = REL_LABEL_MAPPING[arg['role']['type']]
                                relations.append((event_start, arg['startOffset'], relation))                      
                
                if with_relations:
                    # Add 'Same' relation labels to event nugget pairs (identified by start character index)
                    for ix, start_char in enumerate(same_events_char_start[:-1]):
                        relations.append((same_events_char_start[ix], same_events_char_start[ix+1],'Same')) 

            # Add info
            text_info['fname'] = fname
            text_info['text'] = text
            text_info['entities'] = entities
            text_info['relations'] = relations
            data.append(text_info)

    print()
    return data # return list of Doc objs

def create_docbin(data, subset, outdir=''):
    # Based on https://spacy.io/usage/training
    """ Creates Spacy docs with a data split and sets CASIE annotated argument and event nugget 
    labels to .ents attrib (input data to train a Spacy named entity recognizer). 
    It sets also relation labels to the .rel attribute in user_data (accessible as
    Doc._.rel). Saves data to disk in a .spacy file. 
    @ data: list; output of the process_casie function
    @ subset: str; 'train', 'dev' or 'test'.
    @ outdir: str; path to location where the created docbins will be saved
    """
    print('Creating {} subset...'.format(subset))
    Doc.set_extension("rel", default={}, force=True) # register extension attribute
    db = DocBin(store_user_data=True)
    boundary_mismatch_cnt = 0
    ent_cnt = 0
    skipped_files = []
    all_labels = []
    all_rel_labels = []
    nr_rel_issues = 0
    for text_info in data:
        
        # Process text with Spacy 
        text = text_info['text']
        doc = nlp(text) 
        doc.user_data = {'doc_id':text_info['fname'].replace('.json', '')}
        
        # ADD ENTITIES to Spacy Doc
        ents = []
        labels = []
        pos = 0
        for start, end, label in text_info['entities']:
            span = doc.char_span(start, end, label=label)
            if not span:
                boundary_mismatch_cnt += 1
            else:
                labels.append(label)
            ents.append(span)
        ent_cnt += len(ents)
        # Some ents are None (when character boundaries don't align with token boundaries).
        # We remove these to avoid Spacy throwing a TypeError.
        filtered_ents = list(filter(lambda item: item is not None, ents)) 
        # Remove duplicate or overlapping spans
        filtered_ents = filter_spans(filtered_ents)  
        # Exclude documents with no usable span left
        if not len(filtered_ents): 
            skipped_files.append(text_info['fname'])
        # Filter doc.s with many boundary mismatches
        if abs(len(ents) - len(filtered_ents)) > 5: 
            skipped_files.append(text_info['fname'])
        doc.ents = filtered_ents

        if text_info['fname'] not in skipped_files:

            # ADD RELATIONS to Spacy doc 
            # Map start character indices to the corresponding start token index 
            # (required by the relation extraction component)
            start_char_to_tkn_ix = {}
            for span in doc.ents:
                start_char_to_tkn_ix[span.start_char] = span.start
            # Add relation label container for all combinations of start entity token indices 
            rels = {}
            unique_start_tkn_ix_w_rel = set([v for k,v in start_char_to_tkn_ix.items()])
            for x1 in unique_start_tkn_ix_w_rel:
                for x2 in unique_start_tkn_ix_w_rel:
                    rels[(x1, x2)] = {}      
            # Add annotated relation labels to Doc based on the mapped token indices
            for start_char_ent1, start_char_ent2, label in text_info['relations']:
                try:
                    rels[(start_char_to_tkn_ix[start_char_ent1], start_char_to_tkn_ix[start_char_ent2])][label] = 1.0
                    pos += 1
                    # Treat all relations as symmetric by add label also for the entity pair in inverted order 
                    #if label not in rels[(start_char_to_tkn_ix[start_char_ent2], start_char_to_tkn_ix[start_char_ent1])]:
                    #    rels[(start_char_to_tkn_ix[start_char_ent2], start_char_to_tkn_ix[start_char_ent1])][label] = 1.0
                    #    pos += 1
                except KeyError: # catch issues with Spacy token boundaries and character offsets from the relation annotations
                    #print(text_info['fname'], start_char_ent1, start_char_ent2, label)
                    nr_rel_issues += 1
            # Fill in zero's for entity pairs with no relation annotations 
            #mapped_labels = [v for k,v in REL_LABEL_MAPPING.items()]
            for x1 in unique_start_tkn_ix_w_rel:
                for x2 in unique_start_tkn_ix_w_rel:
                    for label in REL_LABEL_MAPPING.values():
                        if label not in rels[(x1, x2)]:
                            rels[(x1, x2)][label] = 0.0
            doc._.rel = rels # will be part of user data (and needs to come after creating it)

            all_labels.extend(labels)
            positive_rel_lbls = [kk for k,v in rels.items() for kk,vv in v.items() if vv]
            all_rel_labels.extend(positive_rel_lbls)

            # Add Doc to Spacy DocBin if it has relations
            if positive_rel_lbls:
                db.add(doc)

    #if boundary_mismatch_cnt:
    #    print('{} ent with boundary mismatch out of a total of {} ents.'.format(boundary_mismatch_cnt, ent_cnt))
    
    # Save dataset files
    if not os.path.isdir(outdir):
        os.mkdir(outdir)  
    db.to_disk(os.path.join(outdir, "./{}.spacy".format(subset)))
    print('Saved {} files (skipped {})'.format(len(data)-len(skipped_files), len(skipped_files)))
    print()
    
    print('Entity label distribution')
    for lbl, cnt in sorted(Counter(all_labels).items()):
        print(cnt, lbl)
    print()

    print('Relation label distribution')
    for lbl, cnt in sorted(Counter(all_rel_labels).items()):
        print(cnt, '\t', lbl)

    print('# NER classes', len(set(all_labels)))
    print('# REL classes', len(set(all_rel_labels)))

    return db

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-data_dir", type=str, help="Path to CASIE 'data/' directory", required=True)
    argparser.add_argument("-eval_size", type=int, help="The size of the development and the test set. (Same size is used.)", default=100)
    argparser.add_argument("-attr_type", type=str, help="The Spacy Doc attribute to which to add annotations 'ents' (to use data with an entity recognizer) or 'spans' (to use data with a span categorizer)", default='ents')
    argparser.add_argument("-span_key", type=str, help="The span key specified in the config file of the span categorizer to be used.", default='sc')
    argparser.add_argument("-task", type=str, help="Type of prediction task, 'event' for classifying event type and realis (actually occured or not); 'arg' for argument type classification or 'all' for both (default)" , default='all')
    argparser.add_argument("-outdir", type=str, help="Directory where to save the created dataset files (overwrites existing files)", default='casie')
    argparser.add_argument("-with_relations", help="Whether to extract also relation annotations", action='store_true')

    args = argparser.parse_args()

    texts_dir = os.path.join(args.data_dir, 'source')
    ann_dir = os.path.join(args.data_dir, 'annotation')

    REL_LABEL_MAPPING = convert_to_unspecified_rel(RELATION_LABELS, REL_LABEL_MAPPING)

    nlp = spacy.blank("en") # Create a blank Tokenizer with just the English vocab
    random.seed(1234)
    data = preprocess_casie(ann_dir, texts_dir, args.task, args.with_relations)
    random.shuffle(data)
    
    train = create_docbin(data[args.eval_size*2:], 'train',  args.outdir)
    dev = create_docbin(data[args.eval_size:args.eval_size*2], 'dev', args.outdir)
    test = create_docbin(data[:args.eval_size], 'test', args.outdir)

    print('Creating {} train, {} dev and {} test instances.'.format(len(train), len(dev), len(test)))
    

