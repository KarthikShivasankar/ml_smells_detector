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
             'Price', 'Purpose', 'Releaser', 'Same', 'Supported_Platform', 'Time', 
             'Tool', 'Trusted-Entity', 'Victim', 'Vulnerability', 'Vulnerable_System', 
             'Vulnerable_System_Owner', 'Vulnerable_System_Version']

RELEVANT_RELATION_LABELS = ['Attacker', 'Releaser',                     # Actor
                            'Victim', 'Vulnerable_System_Owner',        # Asset (Person, Org)
                            'Vulnerable_System', 'Vulnerable_System_Version', 'Compromised-Data', # Asset
                            'Tool', 'Attack-Pattern', 'Capabilities',   # Method
                            'Purpose',                                  # Impact (motivation)
                            'Damage-Amount',                            # Loss 
                            'Same']

Doc.set_extension("rel", default={}, force=True) # register extension attribute

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
    as entity label. 
    Returns a list with tuples of text and the list of entity information tuples.
    @ ann_dir: str; path to the CASIE JSON data with annotations;
    @ texts_dir: str; path to the CASIE text data in 'source/'.
    @ task: 'arg' for argument type classification,
            'event' for event type and realis classification. 
    """
    data = [] 
    #rel_labels = []
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
            relations = {}
            all_events_char_start = []
            for event_cluster in ann['cyberevent']['hopper']: # list of dict with related events ('same')  
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
                    # Use join arg type and binary realis labels
                    #event_type = event_realis + '_' + event_type
                    event_text = event['nugget']['text']
                    event_start = event['nugget']['startOffset']
                    event_end = event['nugget']['endOffset']
                    if task in ['event', 'all']:
                        entities.append((event_start, event_end, event_type))
                    elif task == 'realis':
                        entities.append((event_start, event_end, event_realis))
                    same_events_char_start.append(event_start)

                    # Add argument labels: type, relation
                    if task in ['arg', 'all']:
                        for arg in event.get('argument', []):
                            entities.append((arg['startOffset'], arg['endOffset'], arg['type']))
                        if with_relations:
                            relation = arg['role']['type']
                            if relation in RELEVANT_RELATION_LABELS:
                                relations[(event_start, arg['startOffset'])] = {}
                                relations[(event_start, arg['startOffset'])][relation] = 1.0                        
                    all_events_char_start.append(same_events_char_start)
                
                if with_relations:
                    # Add 'Same' relation labels to event nugget pairs (identified by start character index)
                    for ix, start_char in enumerate(same_events_char_start[:-1]):
                        relations[(same_events_char_start[ix], same_events_char_start[ix+1])] = {} 
                        relations[(same_events_char_start[ix], same_events_char_start[ix+1])]['Same'] = 1.0
                        relations[(same_events_char_start[ix], same_events_char_start[ix+1])]['Diff'] = 0.0
            if with_relations:
                # Add relation labels for all event nugget combinations that are not the same 
                if task == 'event':
                    for p in itertools.permutations(all_events_char_start, 2):
                        non_same_event_combos = zip(*p)
                        for combo in non_same_event_combos:
                            relations[combo] = {}
                            relations[combo]['Same'] = 0.0
                            relations[combo]['Diff'] = 1.0
                else:
                    # Fill in zero's for all label types that are not the annotated relation label
                    for ent_pair in relations:
                        for rel_label in RELEVANT_RELATION_LABELS:
                            if rel_label not in relations[ent_pair]:
                                relations[ent_pair][rel_label] = 0.0
            
            # Add info
            text_info['fname'] = fname
            text_info['text'] = text
            text_info['entities'] = entities
            text_info['relations'] = relations
            
            data.append(text_info)

    print()
    return data

def create_docbin(data, subset, outdir=''):
    # Based on https://spacy.io/usage/training
    """ Creates Spacy docs with a data split and sets CASIE annotated argument and event nugget 
    labels to .ents attrib (input data to train a Spacy named entity recognizer). 
    Saves data to disk in a .spacy file. 
    @ data: list; output of the process_casie function
    @ subset: str; 'train', 'dev' or 'test'.
    """
    print('Creating {} subset...'.format(subset))
    db = DocBin(store_user_data=True)
    boundary_mismatch_cnt = 0
    ent_cnt = 0
    skipped_files = []
    all_labels = []
    all_rel_labels = []
    for text_info in data:
        text = text_info['text']
        doc = nlp(text) # version up to 26 07 2023 19:00 
        #doc.set_extension("rel", default={}, force=True) # register extension attribute
        ents = []
        labels = []
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
        #if len(ents) != len(filtered_ents):
        #    print('\t{}: {} boundary mismatches'.format(text_info['fname'], len(ents) - len(filtered_ents)))
        
        # Filter doc.s with many boundary mismatches
        if abs(len(ents) - len(filtered_ents)) > 5: 
            skipped_files.append(text_info['fname'])
        #try:
        doc.ents = filtered_ents
        doc.user_data = {'doc_id':text_info['fname'].replace('.json', '')}
        doc._.rel = text_info['relations'] # will be part of user data, so rel value assignment needs to come after
        #except ValueError:
        #    print('\t{}: ValueError with token included in more than one span in entities'.format(text_info['fname']))
        #    skipped_files.append(text_info['fname'])
        if text_info['fname'] not in skipped_files:
            db.add(doc)
            all_labels.extend(labels)
            all_rel_labels.extend([kk for k,v in text_info['relations'].items() for kk,vv in v.items() if vv])

    skipped_files = list(set(skipped_files))
    #print(skipped_files)

    # last_doc = list(db.get_docs(nlp.vocab))[-1]
    # print(doc.has_extension('rel'))
    # for rel, v in last_doc._.rel.items():
    #     print(rel, v)

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

    print('Relation label distribution')
    for lbl, cnt in sorted(Counter(all_rel_labels).items()):
        print(cnt, '\t', lbl)

    print('# classes', len(set(all_labels)))

    return db

# def create_docbin_spancat(data, subset, span_key, outdir=''):
#     """ Creates Spacy docs with a data split and sets CASIE annotated argument and event nugget 
#     labels to .spans attrib (input data usable to train a Spacy span categorizer). 
#     Saves data to disk in a .spacy file. 
#     @ data: list; output of the process_casie function
#     @ subset: str; 'train', 'dev' or 'test';
#     @ span_key: str; specified in the config file (chosen by user, usually 'sc') 
#     """
#     print('Creating {} subset...'.format(subset))
#     db = DocBin()
#     all_labels = []
#     for text_info in data:
#         text = text_info['text']
#         doc = nlp(text)
#         ents = []
#         labels = []
#         # Get annotated entities from the corpus and set them to the .ents attrib
#         for start, end, label in text_info['entities']:
#             # fix character offset outside doc (in 464.json)
#             if end > len(text):
#                 print(text_info['fname'], len(text), end)
#                 end = len(text)
#             span = doc.char_span(start, end, label=label)
#             ents.append(span)
#             labels.append(label)
#         all_labels.extend(labels)
#         filtered_ents = list(filter(lambda item: item is not None, ents))
#         filtered_ents = filter_spans(filtered_ents) # deals with overlapping entity indices
#         doc.spans[span_key] = filtered_ents
#         db.add(doc)

#     # Save dataset files
#     if not os.path.isdir(outdir):
#         os.mkdir(outdir)
#     db.to_disk(os.path.join(outdir, "./{}.spacy".format(subset)))
    
#     for lbl, cnt in sorted(Counter(all_labels).items()):
#         print(cnt, lbl)
#     return db

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

    # Create a blank Tokenizer with just the English vocab
    nlp = spacy.blank("en")
    random.seed(1234)
    data = preprocess_casie(ann_dir, texts_dir, args.task, args.with_relations)
    random.shuffle(data)
    
    if args.attr_type == 'spans':
        train = create_docbin_spancat(data[args.eval_size*2:], 'train', args.span_key, args.outdir)
        dev = create_docbin_spancat(data[args.eval_size:args.eval_size*2], 'dev', args.span_key, args.outdir)
        test = create_docbin_spancat(data[:args.eval_size], 'test', args.span_key, args.outdir)
    else:
        train = create_docbin(data[args.eval_size*2:], 'train',  args.outdir)
        dev = create_docbin(data[args.eval_size:args.eval_size*2], 'dev', args.outdir)
        test = create_docbin(data[:args.eval_size], 'test', args.outdir)

    print('Creating {} train, {} dev and {} test instances.'.format(len(train), len(dev), len(test)))
    

