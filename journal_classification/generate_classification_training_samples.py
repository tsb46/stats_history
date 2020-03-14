import argparse
import json
import numpy as np
import pandas as pd
import pickle

from glob import glob


def generate_training_samples(corpus_files, pmid_to_journal, n_samples_per_journal):
    training_samples = []
    for file_path in corpus_files:
        samples = json.load(open(file_path, 'r'))
        sampling_index = np.random.permutation(len(samples))
        for i in range(0, int(n_samples_per_journal)):
            sample = samples[sampling_index[i]]
            # Ensure random selection has a corresponding 'journal' name in
            # pmid to journal dict
            if sample['pmid'] in pmid_to_journal:
                sample_info = {
                    'pmid': sample['pmid'],
                    'title': sample['title'],
                    'abstract': sample['abstract'],
                    'journal': pmid_to_journal[sample['pmid']]['journal']
                }
                training_samples.append(sample_info)
    return training_samples


def run_main(input_dir, output_dir, n_samples_per_journal=200):
    # Get paths to necessary inputs
    pmid_to_journal = json.load(open(input_dir + '/article_metadict.json', 'rb'))
    corpus_files = glob(input_dir + '/preprocessed/*.json')
    # Generate random training samples for labeling
    training_samples = generate_training_samples(corpus_files, pmid_to_journal,
                                                 n_samples_per_journal)
    # Convert to dataframe and output as .csv
    training_samples_df = pd.DataFrame(training_samples)
    training_samples_df.to_csv(output_dir + '/journal_classification_training.csv', index=False)


if __name__ == '__main__':
    """Generate random selections of articles from corpus to 
    label for text classification training"""
    parser = argparse.ArgumentParser(description='Generate random article '
                                                 'selections for labeling')
    parser.add_argument('-d', '--corpus_dir',
                        help='<Required> Path to directory containing '
                             'pubmed corpus files',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write samples to',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_samples_per_file',
                        help='<Required> # of random selections'
                             'per corpus file',
                        required=True,
                        type=str)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['corpus_dir'], args_dict['output_dir'],
             args_dict['n_samples_per_file'])