import argparse
import pandas as pd
import pickle

from glob import glob

def run_main(input_dir, output_dir, domain_classifier, verbose=True):
    domain_classifier = pickle.load(open(domain_classifier, 'rb'))
    ent_files = glob(input_dir + '/*.pickle')
    domain_dict = {}
    for file in ent_files:
        if verbose:
            print(file)
        tmp_dict = pickle.load(open(file, 'rb'))
        article_dicts = {}
        for article in tmp_dict:
            if article is not None:
                if article['parsed'] and len(article['ents']) > 0:
                    article_dicts[article['pmcid']] = {
                                                       'title': article['title'],
                                                       'journal': article['journal'],
                                                       'abstract': article['abstract'],
                                                       }
        article_df = pd.DataFrame.from_dict(article_dicts, orient='index')
        # Free up memory
        del tmp_dict, article_dicts
        if verbose:
            print(f'Predicting article domains for {article_df.shape[0]} articles')
        predicted_domains = domain_classifier.predict(article_df)
        for pmcid, pred_domain in zip(article_df.index, predicted_domains):
            domain_dict[pmcid] = pred_domain

    pickle.dump(domain_dict, open(f'{output_dir}/article_domain_prediction.pickle', "wb"))


if __name__ == '__main__':
    """Extract named entities from pubmed corpus"""
    parser = argparse.ArgumentParser(description='Extract named entities from '
                                                 'pubmed corpus files')
    parser.add_argument('-i', '--input_dir',
                        help='<Required> Path to directory containing '
                             'pubmed xml files',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write outputs (.pickle files) to',
                        required=False,
                        default='results',
                        type=str)
    parser.add_argument('-m', '--classifier_model',
                        help='<Required> path to trained classifier model'
                             ' .pickle file',
                        required=True,
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_dir'], args_dict['output_dir'],
             args_dict['classifier_model']
             )