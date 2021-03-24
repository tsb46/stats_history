import argparse
import json
import numpy as np
import pandas as pd
import pickle

from glob import glob


def generate_training_samples(articles, rand_sample):
    training_samples = []
    for i in rand_sample:
        article = articles[i]
        try:
            sample_info = {
                'pmcid': article['pmcid'],
                'title': article['title'],
                'abstract': article['abstract'],
                'journal': article['journal']
            }
            training_samples.append(sample_info)
        except KeyError:
            pass
    return training_samples

def run_main(input_dir, n_samples=1000):
    # Get paths to necessary inputs
    ent_files = glob(input_dir + '/*.pickle')
    n_samples_per_file = int(np.ceil(n_samples/len(ent_files)))
    training_samples = []
    for file in ent_files:
        articles = article = pickle.load(open(file, 'rb'))
        rand_sample = np.random.permutation(len(articles))[:n_samples_per_file]
        # Generate random training samples for labeling
        training_samples.extend(generate_training_samples(articles, rand_sample))
    # Convert to dataframe and output as .csv
    training_samples_df = pd.DataFrame(training_samples)
    training_samples_df.to_csv('journal_classification/journal_classification_training.csv', 
                               index=False)


if __name__ == '__main__':
    """Generate random selections of articles from corpus to 
    label for text classification training"""
    parser = argparse.ArgumentParser(description='Generate random article '
                                                 'selections for labeling')
    parser.add_argument('-i', '--corpus_dir',
                        help='<Required> Path to directory containing '
                             'parsed pubmed corpus files',
                        required=True,
                        default='results/ents/original',
                        type=str)
    parser.add_argument('-n', '--n_samples',
                        help='<Required> # of random selections'
                             'per corpus file',
                        required=False,
                        default=1000,
                        type=str)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['corpus_dir'], args_dict['n_samples'])
