import json, os
import pytz
import utils 
import argparse
import spacy    
import spacy_fastlang
import calendar
import pandas as pd
from keybert import KeyBERT
from Levenshtein import distance
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter

nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words |= {"com"}
nlp.Defaults.stop_words |= {d.lower() for d in calendar.day_name}
nlp.Defaults.stop_words |= {m.lower() for m in calendar.month_name}
nlp.Defaults.stop_words |= {'thumbnail', 'png', 'jpeg'}
nlp.add_pipe("language_detector")

def postprocess_keywords(keywords, sim_threshold):
    """ Convert keywords to baseform, remove duplicates and keywords containing 
    stopwords and some close class POS tags.
    Tagset: https://universaldependencies.org/u/pos/
    """
    pos_to_filter = ['ADP', 'AUX', 'SYM', 'ADV']
    proc_keywords = []
    for kw, score in keywords:
        if score >= sim_threshold:
            # remove keywords containing closed POS class words
            kw_tkns = nlp(kw)
            for t in kw_tkns: 
                if t.pos_ in pos_to_filter: 
                    break
            else:
                # check if new keyword differs with at least one character or one word compared to any other keywords found
                lemmatized_kw = ' '.join([t.lemma_ for t in kw_tkns])
                lemmatized_kw = correct_lemmatization(lemmatized_kw)
                for detected_kw, sc in proc_keywords:
                    if distance(lemmatized_kw, detected_kw) < 2 or distance(sorted(lemmatized_kw.split()), sorted(detected_kw.split())) < 2:
                        break
                else:
                    proc_keywords.append((lemmatized_kw, score))
    return proc_keywords
    
def get_keywords(texts, sim_threshold, keyword_f='keywords.csv'):
    """Get keywords sorted by highest similarity.
    """
    with open('seed_keywords_cleaned.txt') as f:
        seed_keywords = [l.strip() for l in f.readlines()]
    print('Extracting keywords...')
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(texts, keyphrase_ngram_range=(1, 3),
                                         #stop_words=list(nlp.Defaults.stop_words), # results in keywords with discontinuous token sequences
                                         #seed_keywords=seed_keywords,
                                         min_df=1, # 1 by default
                                         #use_maxsum=True, nr_candidates=20, top_n=5,
                                         use_mmr=True, diversity=0.6) # diversify results
    flattened_kws = sum(keywords, [])
    proc_keywords = postprocess_keywords(sorted(flattened_kws, key=lambda x: x[1], reverse=True), 
                                         sim_threshold)
    for kw, score in proc_keywords[:20]:
        print(round(score, 2,), '\t', kw)
    df = pd.DataFrame(proc_keywords, columns =['keyword', 'score'])
    df.to_csv(keyword_f, encoding='utf-8')

    # Save keywords
    #with open('keywords.txt', 'w') as f:
    #    f.write('\n'.join([str(score) + '\t'+ kw for (kw, score) in proc_keywords  ]))

def remove_similar_text(texts):
    """ Remove potentially duplicate texts from the collection.
    """
    pass

def correct_lemmatization(proc_text):
    "Correct common lemmatization errors."
    if ' datum ' in proc_text:
        proc_text = proc_text.replace(' datum ', ' data ')
    return proc_text

def get_data_stats(texts, include_tokens=False):
    print("Nr texts:", len(texts))
    print("Nr unique texts:", len(set(texts)))
    if include_tokens:
        nr_tokens = 0
        for ix, text in enumerate(texts):
            proc_text = nlp(text)
            nr_tokens += len(proc_text)
        print("Nr tokens:", nr_tokens)
        
def count_keywords(texts, keywords, normalize=False):
    """ Compute keyword frequency in texts. Multiple occurrences
    within the same document are considered only once. 
    @ normalize: divide occurrence counts by the number of texts
    """
    kw_count = {}
    for text in texts: 
        if len(text) > 1000000:
            text = text[:1000000]
        proc_text = ' '.join([t.lemma_ for t in nlp(text)])
        proc_text = correct_lemmatization(proc_text)
        for kw in keywords:
            if kw in proc_text:
                kw_count[kw] = kw_count.get(kw, 0) + 1
    if normalize:
        kw_count = {kw : frq/len(texts) for (kw,frq) in kw_count.items()}
    for kw, freq in sorted(kw_count.items(), key=lambda x: x[1], reverse=True)[:30]:
        print(kw, round(freq, 3))
    return kw_count

def check_lang(texts):
    "Check the language of the documents. "
    lgs = []
    unique_texts = list(set([text if len(text) < 1000000 else text[:1000000] for text in texts]))
    print('# unique texts: ', len(unique_texts))
    print('Detecting languages ...')
    docs = [nlp(text) for text in unique_texts]
    for doc in docs:
        if doc._.language_score >= 0.8:
            lgs.append(doc._.language)
        else:
            lgs.append('unknown')
    lg_cnt = Counter(lgs)
    for lgs, cnt in lg_cnt.most_common():
        print(cnt, round(cnt/len(unique_texts)*100,2), lgs)

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-data_dir", type=str, help="Path to the zip file with SCIO data.", required=True)
    argparser.add_argument("-start_date", type=str, help="Start date for crawled data to use in YYYY-MM-DD format.", required=True)
    argparser.add_argument("-end_date", type=str, help="End date for crawled data to use in YYYY-MM-DD format.", required=True)
    argparser.add_argument("-interval", type=int, help="Number of days to use as time interval (the max. is a month).", default=0)
    argparser.add_argument('-save_rel_docs', help="Save relevant docs used for the keyword extraction.", action='store_true')
    argparser.add_argument('-skip_extraction', help="Skip keyword extraction.", action='store_true')
    argparser.add_argument('-detect_lg', help="Run language language detection.", action='store_true')
    args = argparser.parse_args()

    # Parse and localize dates
    utc = pytz.UTC
    relevant_docs_fn = 'scio_data_%s_%s.json' % (args.start_date, args.end_date)
    start_date = utc.localize(datetime.strptime(args.start_date, '%Y-%m-%d'))
    end_date = utc.localize(datetime.strptime(args.end_date, '%Y-%m-%d'))
    
    # Load relevant texts
    doc_objects = utils.load_scio_texts(args.data_dir, start_date, end_date)
    if args.save_rel_docs:
        with open(relevant_docs_fn, 'w') as fp:
            json.dump(doc_objects, fp)
    texts = [doc_obj['text'] for doc_obj in doc_objects]
    get_data_stats(texts)

    # Extract keywords for total period
    kw_out_f = 'keywords_' + str(start_date.date()) + '_' + str(end_date.date()) + '.csv'
    if not args.skip_extraction:
        get_keywords(texts, 0.5, keyword_f=kw_out_f)
    
    # Count keywords in data subsets of the chosen time interval
    if args.interval:
        kws_df = pd.read_csv(kw_out_f)
        kws = kws_df['keyword'].tolist()
        if args.interval >= 30: 
            time_step = relativedelta(months=+1)
        else:
            time_step = timedelta(days=nr_days)
        print(start_date.date(), end_date.date())
        while start_date < end_date:
            print(start_date.date(), start_date + time_step)
            texts_subset = []
            for doc_obj in doc_objects:
                doc_date = utc.localize(datetime.strptime(doc_obj['date'], '%Y-%m-%d'))
                if doc_date >= start_date and doc_date < start_date + time_step: 
                    if doc_obj['text'] not in texts_subset: # avoid duplicate texts
                        texts_subset.append(doc_obj['text'])
            count_keywords(texts_subset, kws, normalize=True)
            start_date = start_date + time_step
            print()

    # Check the language of documents
    if args.detect_lg:
        check_lang(texts)

