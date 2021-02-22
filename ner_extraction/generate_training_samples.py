import json
import argparse
import os
import pickle
import numpy as np
import spacy

from spacy.matcher import PhraseMatcher
from trainer_utils import clean_out_html_tags, \
    replace_unwanted_characters, filter_matches, pull_xml_fps, \
    parse_pubmed_article

np.random.seed(0)

def format_output_spacy(doc, match_indices, entity_label):
    output = (str(doc),
              {
                  'entities': [
                      (doc[start:end].start_char, doc[start:end].end_char, entity_label)
                      for _, start, end in match_indices
                  ]
              }
              )
    return output


def run_main(input_dir, output_dir,
             n_samples, type_label, ent_thres,
             phrase_file, section_keyword_file,
             verbose):
    # Set up inputs necessary to run main function
    input_dict = set_up_inputs(input_dir, output_dir, type_label,
                               phrase_file, section_keyword_file, 
                               ent_thres, n_samples, verbose)
    # Initialize empty training samples list
    training_samples = []
    # Initialize loop count, and continue while loop until # of samples is complete
    loop_count = 0
    while len(training_samples) < n_samples:
        # Run a training instance - i.e. get a single training sample
        training_sample = run_training_instance(
            input_dict['files'][input_dict['sampling_index'][loop_count]],
            input_dict['spacy_model'],
            input_dict['phrase_matcher_case'],
            input_dict['phrase_matcher_nocase'],
            input_dict['section_keywords'],
            input_dict['ent_types'],
            input_dict['ent_thres']
        )
        if training_sample is not None:
            training_samples.append(training_sample)
            if verbose:
                print(
                    '# of training samples created: {}'.format(len(training_samples))
                )
        loop_count += 1
    write_output(training_samples, input_dir, output_dir)


def run_training_instance(xml_fp, spacy_model, phrase_matcher_case, 
                          phrase_matcher_nocase, section_keywords,
                          ent_label, ent_thres):
    xml_dict = parse_pubmed_article(xml_fp, section_keywords)
    # Standardize text
    if xml_dict is not None:
        text = standardize_text(xml_dict['section_text'])
        # Run spacy model on doc
        text_spacy = spacy_model(text)
        matches_case = phrase_matcher_case(text_spacy)
        matches_nocase = phrase_matcher_nocase(text_spacy)
        matches_all = matches_case + matches_nocase
        if len(matches_all) > 0:
            # Filter out overlapping spans (take the longest one)
            matches_filtered = filter_matches(text_spacy, matches_all)
            # if greater than entity threshold, format for output
            if len(matches_filtered) >= ent_thres:
                training_sample = format_output_spacy(text_spacy, matches_filtered, ent_label)
                return training_sample


def seed_phrase_casesplit(search_phrases):
    search_phrase_lower = [seed for seed in search_phrases 
                           if seed.islower()]
    search_phrase_case = [seed for seed in search_phrases 
                          if seed not in search_phrase_lower]
    return search_phrase_lower, search_phrase_case


def set_up_inputs(input_dir, output_dir, type_label,
                  phrase_file, section_keyword_file,
                  ent_thres, n_samples, verbose):
    # Pull text from json into one list
    xml_fps = pull_xml_fps(input_dir)
    if verbose:
        print(f'Generating {n_samples} training samples from {len(xml_fps)} pubmed articles')
    # Get entity type labels from delimited string input
    ent_type = type_label
    # take a random permutation of all files to randomly sample
    sampling_index = np.random.permutation(len(xml_fps) - 1)
    # Load 'en_core_web_md' spacy model
    spacy_model = spacy.load('en_core_web_sm')
    # Set phrase seeds to guide selection of sentences
    with open(phrase_file) as f:
        search_phrases = f.read().splitlines()
    # Set section keywords to guide selection of article sections
    with open(section_keyword_file) as f:
        section_keywords = f.read().splitlines()
    # Remove potential empty lines and white space
    search_phrases = [seed.strip() for seed in search_phrases if seed != '']
    section_keywords = [keyword.strip() for keyword in section_keywords if keyword != '']
    # Initialize PhraseMatcher obj for case-sensitive and non-case-sensitive matching
    matcher_case = PhraseMatcher(spacy_model.vocab)
    matcher_nocase = PhraseMatcher(spacy_model.vocab, attr='LOWER')
    # Create phrase dictionary and add to matcher
    search_phrases_lw, search_phrases_case = seed_phrase_casesplit(search_phrases)
    patterns_lw = [spacy_model.make_doc(text) for text in search_phrases_lw]
    patterns_case = [spacy_model.make_doc(text) for text in search_phrases_case]
    matcher_nocase.add(ent_type[0], None, *patterns_lw)
    matcher_case.add(ent_type[0], None, *patterns_case)
    input_dict = {
        'files': xml_fps,
        'sampling_index': sampling_index,
        'ent_types': ent_type,
        'phrase_matcher_case': matcher_case,
        'phrase_matcher_nocase': matcher_nocase,
        'section_keywords': section_keywords,
        'ent_thres': int(ent_thres),
        'spacy_model': spacy_model,
        'output_dir': output_dir
    }
    return input_dict


def standardize_text(doc):
    # Standardize text
    doc = clean_out_html_tags(doc)
    doc = replace_unwanted_characters(doc)
    return doc


def write_output(training_samples, input_dir, output_dir):
    output_dict = {'training_samples': training_samples}
    input_base = os.path.basename(input_dir)
    input_name = os.path.splitext(input_base)[0]
    pickle.dump(output_dict,
                open('{}/training_samples_{}.pickle'.format(output_dir,
                                                            input_name), "wb"))


if __name__ == '__main__':
    """Retrieve list of text data and start CL interface to create 
    training examples for Named Entity Recognition model"""
    parser = argparse.ArgumentParser(description='Create training examples '
                                                 'for NER model')
    parser.add_argument('-i', '--input_dir',
                        help='<Required> xml file directory',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_samples',
                        help='<Required> num of training samples to'
                             'pull from text',
                        required=True,
                        type=int)
    parser.add_argument('-e', '--entity_label',
                        help='<Required> delimited list string of the '
                             'names of the entity types you are '
                               'training the NER model for',
                        required=True,
                        type=str)
    parser.add_argument('-s', '--search_phrases',
                        help='Path to .txt file providing a set of phrase strings'
                               'to guide selection of training samples. The .txt'
                               'file should have each string on a separate line',
                        required=True,
                        type=str)
    parser.add_argument('-k', '--section_keywords',
                        help='Path to .txt file providing a set of keywords'
                               'to guide selection of text in article xml. The .txt'
                               'file should have each string on a separate line',
                        required=True,
                        type=str)
    parser.add_argument('-t', '--ent_threshold',
                        help='Set the minimum number of entities that must be '
                             'discovered in a document to be included as a training'
                             'sample',
                        required=False,
                        default=1,
                        type=str),
    parser.add_argument('-v', '--verbosity',
                        required=False,
                        default='false',
                        type=str),
    parser.add_argument('-o', '--output_dir',
                        help='Path to directory '
                             'to write output pickle file to',
                        required=False,
                        default='results',
                        type=str)

    args_dict = vars(parser.parse_args())
    if args_dict['verbosity'] == 'true':
        verbose = True
    else:
        verbose = False
    run_main(args_dict['input_dir'], args_dict['output_dir'],
             args_dict['n_samples'],
             args_dict['entity_label'],
             args_dict['ent_threshold'],
             args_dict['search_phrases'],
             args_dict['section_keywords'],
             verbose)
