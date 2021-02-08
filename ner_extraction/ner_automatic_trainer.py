import json
import argparse
import os
import pickle
import numpy as np
import spacy

from spacy.matcher import PhraseMatcher
from ner_trainer.trainer_utils import clean_out_html_tags, \
    replace_unwanted_characters, filter_matches


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


def run_main(input, output_dir,
             n_samples, type_label, ent_thres,
             phrase_file, verbose=False):
    # Set up inputs necessary to run main function
    input_dict = set_up_inputs(input, output_dir, type_label,
                               phrase_file, ent_thres, n_samples)
    # Initialize empty training samples list
    training_samples = []
    # Initialize loop count, and continue while loop until # of samples is complete
    loop_count = 0
    while len(training_samples) < n_samples:
        # Run a training instance - i.e. get a single training sample
        training_sample = run_training_instance(
            input_dict['text'][input_dict['sampling_index'][loop_count]],
            input_dict['spacy_model'],
            input_dict['phrase_matcher'],
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
    write_output(training_samples, input, output_dir)


def run_training_instance(text, spacy_model, phrase_matcher,
                          ent_label, ent_thres):
    # Standardize text
    text = standardize_text(text)
    # Run spacy model on doc
    text_spacy = spacy_model(text)
    matches = phrase_matcher(text_spacy)
    if len(matches) > 0:
        # Filter out overlapping spans (take the longest one)
        matches_filtered = filter_matches(text_spacy, matches)
        # if greater than entity threshold, format for output
        if len(matches_filtered) > ent_thres:
            training_sample = format_output_spacy(text_spacy, matches_filtered, ent_label)
            return training_sample


def set_up_inputs(input, output_dir, type_label,
                  phrase_file, ent_thres, n_samples):
    # Ensure the user supplies a json file as input
    _, ext = os.path.splitext(input)
    if ext == '.json':
        text_json = json.load(open(input, 'r'))
    else:
        raise Exception('.json must be supplied as input')

    # Pull text from json into one list
    text_list = [text['methods'] for text in text_json]
    # Get entity type labels from delimited string input
    ent_type = type_label
    # simply take a permutation of all rows to randomly sample
    sampling_index = np.random.permutation(len(text_list) - 1)
    # Load 'en_core_web_md' spacy model
    spacy_model = spacy.load('en_core_web_sm')
    # Set phrase seeds to guide selection of sentences
    with open(phrase_file) as f:
        search_phrases = f.read().splitlines()
    # Remove potential empty lines
    search_phrases = [seed for seed in search_phrases if seed != '']
    # Initialize PhraseMatcher obj
    matcher = PhraseMatcher(spacy_model.vocab, attr='LOWER')
    # Create phrase dictionary and add to matcher
    patterns = [spacy_model.make_doc(text) for text in search_phrases]
    matcher.add(ent_type[0], None, *patterns)
    input_dict = {
        'text': text_list,
        'sampling_index': sampling_index,
        'ent_types': ent_type,
        'phrase_matcher': matcher,
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


def write_output(training_samples, input, output_dir):
    output_dict = {'training_samples': training_samples}
    input_base = os.path.basename(input)
    input_name = os.path.splitext(input_base)[0]
    pickle.dump(output_dict,
                open('{}/training_samples_{}.pickle'.format(output_dir,
                                                            input_name), "wb"))


if __name__ == '__main__':
    """Retrieve list of text data and start CL interface to create 
    training examples for Named Entity Recognition model"""
    parser = argparse.ArgumentParser(description='Create training examples '
                                                 'for NER model')
    parser.add_argument('-i', '--input',
                        help='<Required> Path to .json file '
                             'containing training text OR path to'
                             'existing training model file',
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
                        required=False,
                        default='none',
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
                        default='data/training_samples',
                        type=str)

    args_dict = vars(parser.parse_args())
    if args_dict['verbosity'] == 'true':
        verbose = True
    else:
        verbose = False
    run_main(args_dict['input'], args_dict['output_dir'],
             args_dict['n_samples'],
             args_dict['entity_label'],
             args_dict['ent_threshold'],
             args_dict['search_phrases'],
             verbose)
