import argparse
import pickle

from glob import glob
from pathlib import Path
from preprocessing import EntityPreprocessing


def article_preprocess(entity_fps, output_dir, ent_preproc, 
                       domain_dict, verbose=True):
    new_fp_list = []
    for file in entity_fps:
        if verbose:
            print(file)
        article_dict = pickle.load(open(file, 'rb'))
        article_proc_dict = ent_preproc.preprocess_articles(article_dict)
        file_stem = Path(file).stem
        new_fp = f'{output_dir}/{file_stem}_proc1.pickle'
        new_fp_list.append(new_fp)
        pickle.dump(article_proc_dict, open(new_fp, "wb"))
    return new_fp_list


def corpus_preprocess(entity_fps, output_dir, ent_preproc, verbose=True):
    full_article_dict = {}
    for file in entity_fps:
        article_dict = pickle.load(open(file, 'rb'))
        full_article_dict.update(article_dict)
    final_article_dict = ent_preproc.preprocess_corpus(full_article_dict)
    pickle.dump(final_article_dict, open(f'{output_dir}/article_ents_final.pickle', "wb"))


def run_main(input_dir, output_dir, domain_dict_fp, n_thres, verbose=True):
    domain_dict = pickle.load(open(domain_dict_fp, 'rb'))
    ent_files = glob(input_dir + '/*.pickle')
    ent_preproc = EntityPreprocessing(domain_dict, n_thres)
    if verbose:
        print('Preprocessing of entity strings within articles')
    proc_fps = article_preprocess(ent_files, output_dir, ent_preproc, domain_dict)
    if verbose:
        print('Preprocessing of entity strings based on corpus statistics')
    corpus_preprocess(proc_fps, output_dir, ent_preproc)


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
    parser.add_argument('-t', '--ent_count_threshold',
                        help='<Required> # of number of instances across corpus an entity '
                        'must have to be included in final entity list',
                        required=False,
                        default=5,
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_dir'], args_dict['output_dir'],
             args_dict['classifier_model'], args_dict['ent_count_threshold']
             )