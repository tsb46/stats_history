import argparse
import json
import pickle

from trainer_utils import pull_xml_fps, parse_pubmed_article


def pull_ents(xml_fp, ner_model, section_keywords):
    try:
        xml_dict = parse_pubmed_article(xml_fp, section_keywords)
    except TypeError:
        return None
    if xml_dict['parsed']:
        try:
            spacy_obj = ner_model(xml_dict['section_text'])
            ents = [str(ent) for ent in spacy_obj.ents]
        except AttributeError:
            ents = []
        xml_dict.update({'ents': ents})
    return xml_dict


def run_main(input_dir, output_dir, spacy_model, section_keyword_file, 
             start_indx, end_indx, verbose=True):
    # Get paths to necessary inputs
    xml_fps = pull_xml_fps(input_dir)
    ner_model = pickle.load(open(spacy_model, 'rb'))
    with open(section_keyword_file) as f:
        section_keywords = f.read().splitlines()
    # Remove potential empty lines and white space
    section_keywords = [keyword.strip() for keyword in section_keywords if keyword != '']
    if end_indx is None:
        end_indx = len(xml_fps)
    xml_fps_batch = xml_fps[start_indx:end_indx]
    parsed_articles = []
    for i, fp in enumerate(xml_fps_batch):
        if verbose and i % 100 == 0:
            print(f'# of parsed articles: {i}')
        article_dict = pull_ents(fp, ner_model, section_keywords)
        parsed_articles.append(article_dict)
    pickle.dump(parsed_articles,
                open(f'{output_dir}/article_ents_{start_indx}_{end_indx}.pickle', "wb"))


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
    parser.add_argument('-m', '--spacy_ner_model',
                        help='<Required> path to trained spacy ner '
                             'model .pickle file',
                        required=True,
                        type=str)
    parser.add_argument('-k', '--section_keywords',
                    help='Path to .txt file providing a set of keywords'
                           'to guide selection of text in article xml. The .txt'
                           'file should have each string on a separate line',
                    required=True,
                    type=str)
    parser.add_argument('-s', '--start_indx',
                        help='(Parsed articles are saved in batches) '
                        'Starting indx of batch',
                        default=0,
                        type=int)
    parser.add_argument('-e', '--end_indx',
                        help='(Parsed articles are saved in batches) '
                        'ending indx of batch - default: # of all pubmed articles',
                        default=None,
                        type=int)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_dir'], args_dict['output_dir'],
             args_dict['spacy_ner_model'],
             args_dict['section_keywords'],
             args_dict['start_indx'],
             args_dict['end_indx']
             )
