import spacy
import random
import argparse
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-ner_path", type=str, help="Path to trained entity recognition (NER) model", required=True)
    argparser.add_argument("-rel_path", type=str, help="Path to trained relation extraction (REL) model", required=True)
    argparser.add_argument("-threshold", type=float, help=" probability threshold to count labels as positive", default=0.3)
    argparser.add_argument("-test_data", type=str, help="Path to a text file (or directly some text) to test extraction.", required=True)
    argparser.add_argument("-from_file", help="Add arg if a text file path is passed in 'test_data', omit it if the text passed directly in 'test_data'", action='store_true')
    args = argparser.parse_args()

    ner_model = spacy.load(args.ner_path)
    rel_model = spacy.load(args.rel_path)
    Doc.set_extension("rel", default={}, force=True) # register extension attribute
    
    if args.from_file:
        # TO DO: implement loading multiple documents
        print("Loading from file not yet implemented, pass a single text instead on the command line with -test_data for a demo.")
    else:
        docs = [args.test_data]
    
    docs_w_ner = ner_model.pipe(docs, disable=["tagger"])
    for doc in docs_w_ner:
        print(doc)
        for e in doc.ents:
            print(e.start, e.text, e.label_)
        print()
        # Apply relation extractor pipeline to the docs with entities detected with the NER pipeline  
        for name, proc in rel_model.pipeline:
            doc = proc(doc)
            #print(doc.user_data)
            # Apply the relation extraction for each pair of entities found in each sentence.
            for ent_tkn_ix_pair, rel_dict in doc._.rel.items():
                for ent1 in doc.ents:
                    for ent2 in doc.ents:
                        if ent1.start == ent_tkn_ix_pair[0] and ent2.start == ent_tkn_ix_pair[1]:
                            #for rel_type in rel_dict:   
                            top_rel = max(rel_dict, key = lambda x: rel_dict[x])
                            if rel_dict[top_rel] >= args.threshold:
                                print(f"{ent1.text} ({ent1.label_}) - {ent2.text} ({ent2.label_}) --> relation: {top_rel} ({rel_dict[top_rel]:.2f})")
        print()