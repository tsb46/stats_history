import json
import argparse
import os
import pickle
import numpy as np
import spacy

from PyInquirer import prompt
from spacy.matcher import PhraseMatcher

from ner_trainer.user_prompts import entity_question, overwrite_pickle
from ner_trainer.trainer_utils import clean_out_html_tags, \
    replace_unwanted_characters, filter_matches


def analyze_chosen_sample(doc, matches, input_dict):
    progress = 1
    while progress == 1:
        print('{}\n'.format(doc))
        for match_id, start, end in matches:
            span = doc[start:end]
            print('Start:End {}:{}  Text: {}'.format(start, end, span.text))
        answer_1 = prompt(entity_question)
        if answer_1['entity_check'].lower() == 'y':
            training_output = format_output_spacy(doc, matches,
                                                  input_dict['ent_types'])
            return training_output
        # If the response was for a progress report
        elif answer_1['entity_check'].lower() == 'p':
            generate_progress_report(input_dict['progress_report'])
            progress = 1
        elif answer_1['entity_check'].lower() == 's':
            write_output(input_dict['training_samples'], input_dict['output_dir'],
                         input_dict['progress_report'])
            progress = 1
        else:
            return


def format_output_spacy(doc, match_indices, entity_label):
    output = (doc,
              {
                  'entities': [
                      (start, end, entity_label)
                      for _, start, end in match_indices
                  ]
              }
              )
    return output


def generate_progress_report(progress_report):
    print('Progress Report: \n')
    if len(progress_report) == 0:
        print('No progress to report :( \n')
    else:
        print('Number of samples generated: {} \n'.format(
            progress_report['n_samples']
        ))
        print('Number of sentences viewed: {} \n'.format(
            progress_report['n_tries']
        ))


def load_pickle_data(pickle_flag, output_dir):
    # If flagged, load in existing training file
    output_pickle = '{}/training_samples.pickle'.format(output_dir)
    if pickle_flag == 'true':
        previous_run = pickle.load(open(output_pickle, "rb"))
        progress_report = previous_run['metadata']
        training_samples = previous_run['training_samples']
    else:
        # if training pickle file exists, and user didn't specify, ask them if
        # they want to load it in, rather than overwrite
        if os.path.isfile(output_pickle):
            answer = prompt(overwrite_pickle)
            if answer['overwrite_pickle']:
                previous_run = pickle.load(open(output_pickle, "rb"))
                progress_report = previous_run['metadata']
                training_samples = previous_run['training_samples']
            else:
                training_samples = []  # initialize list that we add training samples to
                progress_report = {}
        else:
            training_samples = []  # initialize list that we add training samples to
            progress_report = {}
    return progress_report, training_samples


def run_main(input, output_dir,
             n_samples, type_labels,
             phrase_file, pickle_file='false'):
    # Set up inputs necessary to run main function
    input_dict = set_up_inputs(input, output_dir, type_labels,
                               phrase_file, pickle_file)
    # Start the training loop with a while loop that continues
    # until the number of specified training samples.
    if 'n_tries' in input_dict['progress_report'].keys():
        loop_count = input_dict['progress_report']['n_tries']
    else:
        loop_count = 0  # Initialize loop counter
    # Start sampling at top level of hierarchy, if any
    while len(input_dict['training_samples']) < n_samples:
        # Run a training instance - i.e. get a single training sample
        training_sample = run_training_instance(input_dict, loop_count)
        if training_sample is not None:
            input_dict['training_samples'].append(training_sample)
            input_dict['progress_report'] = update_progress_report(
                input_dict['progress_report'],
                loop_count,
                input_dict['training_samples'],
                input_dict['sampling_index']
            )
        loop_count += 1
    write_output(input_dict['training_samples'], input, output_dir,
                 input_dict['progress_report'])


def run_training_instance(input_dict, loop_count):
    # Standardize text
    text = standardize_text(input_dict['text'][input_dict['sampling_index'][loop_count]])
    # Run spacy model on doc
    spacy_model = input_dict['spacy_model']
    phrase_matcher = input_dict['phrase_matcher']
    text_spacy = spacy_model(text)
    matches = phrase_matcher(text_spacy)
    if len(matches) > 0:
        # Filter out overlapping spans (take the longest one)
        matches_filtered = filter_matches(text_spacy, matches)
        # analyze the chosen sample for any entities
        training_sample = analyze_chosen_sample(text_spacy, matches_filtered,
                                                input_dict)
        return training_sample


def set_up_inputs(input, output_dir, type_labels,
                  phrase_file, pickle_file='false'):
    # Ensure the user supplies a json file as input
    _, ext = os.path.splitext(input)
    if ext == '.json':
        text_json = json.load(open(input, 'r'))
    else:
        raise Exception('.json must be supplied as input')

    # Pull text from json into one list
    text_list = [text['methods'] for text in text_json]
    # If flagged, or already exists, get previous runs from pickle file
    progress_report, training_samples = load_pickle_data(pickle_file, output_dir)
    # Get entity type labels from delimited string input
    ent_types = [ent_type for ent_type in type_labels.split(',')]
    # simply take a permutation of all rows to randomly sample
    # create sampling index, if no previous run was loaded
    if 'sampling_index' not in progress_report.keys():
        sampling_index = np.random.permutation(len(text_list) - 1)
    else:
        sampling_index = progress_report['sampling_index']
    # Load 'en_core_web_md' spacy model
    spacy_model = spacy.load('en_core_web_sm')
    spacy_model.tokenizer.token_match = None
    # Set phrase seeds to guide selection of sentences
    with open(phrase_file) as f:
        search_phrases = f.read().splitlines()
    # Remove potential empty lines
    search_phrases = [seed for seed in search_phrases if seed != '']
    # Initialize PhraseMatcher obj
    matcher = PhraseMatcher(spacy_model.vocab, attr='LOWER')
    # Create phrase dictionary and add to matcher
    patterns = [spacy_model.make_doc(text) for text in search_phrases]
    matcher.add(ent_types[0], None, *patterns)
    input_dict = {
        'text': text_list,
        'training_samples': training_samples,
        'sampling_index': sampling_index,
        'ent_types': ent_types,
        'phrase_matcher': matcher,
        'progress_report': progress_report,
        'spacy_model': spacy_model,
        'output_dir': output_dir
    }
    return input_dict


def standardize_text(doc):
    # Standardize text
    doc = clean_out_html_tags(doc)
    doc = replace_unwanted_characters(doc)
    return doc


def update_progress_report(progress_report, loop_count, training_samples,
                           sampling_index):
    progress_report.update(
        {
            'n_tries': loop_count,
            'n_samples': len(training_samples),
            'sampling_index': sampling_index
        }
    )
    return progress_report


def write_output(training_samples, input, output_dir, progress_report):
    output_dict = {'metadata': progress_report,
                   'training_samples': training_samples}
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
    parser.add_argument('-p', '--pickle_file',
                        help='If pre-trained existing file exists in '
                               'output directory, flag it here w/ "true',
                        required=False,
                        default='false',
                        choices=['true', 'false'],
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='Path to directory '
                             'to write output pickle file to',
                        required=False,
                        default='data/training_samples',
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input'], args_dict['output_dir'],
             args_dict['n_samples'],
             args_dict['entity_label'],
             args_dict['search_phrases'],
             args_dict['pickle_file'])
