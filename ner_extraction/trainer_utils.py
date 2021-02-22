import glob
import os
import pubmed_parser as pp
import re

from spacy.util import filter_spans


def build_pubmed_dict(metadata, body, section_keywords):
    """
    Build pubmed dictionary that contains all relevant
    information
    :param metadata: metadata xml parse list
    :param body: article body xml parse list
    :return: dictionary of relevant info for article
    """
    section_text, section_titles = pull_xml_section_txt(body, section_keywords)
    pubmed_dict = {
        'title': metadata['full_title'],
        'abstract': metadata['abstract'],
        'parsed': True,
        'pmid': metadata['pmid'],
        'doi': metadata['doi'],
        'journal': metadata['journal'],
        'year': metadata['publication_year'],
        'date': metadata['publication_date'],
        'section_titles': section_titles,
        'section_text': section_text
    }
    return pubmed_dict


def check_for_section(pubmed_xml, section_keywords):
    """
    Check whether parsed pubmed xml output contains a 'methods' section
    :param pubmed_xml:  pubmed xml - list of dicts output from
    'parse_pubmed_paragrah'
    :return: boolean - 0 (false) or 1 (true)
    """
    section_list = [dict_info['section'] for dict_info in pubmed_xml
                    if dict_info['section'] != '']
    section_list = [dict_info['section'] for dict_info in pubmed_xml
                    if any(marker in dict_info['section'].lower()
                           for marker in section_keywords)
                    if dict_info['section'] != '']
    return len(section_list) > 0


def clean_out_html_tags(text):
    # This function cleans out any HTML tags in a list of
    # texts, or single text.
    tag_re = re.compile(r'<[^>]+>')
    if isinstance(text, list):
        result = [
          tag_re.sub('', abstract).replace('\n',' ') for abstract in
          text
        ]
    else:
        result = tag_re.sub('', text)
    return result


def filter_matches(text_spacy, matches):
    spans_orig = [text_spacy[start:end] for _, start, end in matches]
    spans_filtered = filter_spans(spans_orig)
    match_filter_indx = [index for index, item in enumerate(spans_orig)
                         if item in spans_filtered]
    matches_filtered = [matches[indx] for indx in match_filter_indx]
    return matches_filtered


def parse_pubmed_article(xml_path, section_keywords):
    """
    Parse pubmed xml file into human-readable format

    :param xml_path: path to xml file to parse
    :param section_keywords: keyword strings to filter pubmed article sections
    :return: dictionary file containing relevant fields
    """
    body = pp.parse_pubmed_paragraph(xml_path)
    if check_for_section(body, section_keywords):
        metadata = pp.parse_pubmed_xml(xml_path)
        pubmed_dict = build_pubmed_dict(metadata, body, section_keywords)
        return pubmed_dict
    else:
        metadata = pp.parse_pubmed_xml(xml_path)
        return {
                'pmid': metadata['pmid'],
                'title': metadata['full_title'],
                'journal': metadata['journal'],
                'parsed': False
                }


def pull_xml_fps(input_dir):
    files = []
    for p, d, f in os.walk(input_dir):
        for file in f:
            if file.endswith('xml'):
                files.append(p + '/' + file)
    return files


def pull_xml_section_txt(body, section_keywords):
    """
    Pull section text from parsed xml article
    :param body: parsed text of the 'body' of the article
    :param section_keywords: strings containing keywords for detection of section text
    :return: string of full method section text
    """
    section_indx_title = [(index, line['section']) for index, line in enumerate(body) 
                          if any(marker in line['section'].lower() for marker in section_keywords)
                          if line['section'] != '']
    section_indx = [section[0] for section in section_indx_title]
    section_titles = [section[1] for section in section_indx_title]
    section_text = ' '.join([line['text'] for index, line in enumerate(body)
                           if index in section_indx])
    return section_text, section_titles


def replace_unwanted_characters(text):
    unwanted_strings = ["'", "’", "/"]
    for char in unwanted_strings:
        text = text.replace(char, '')
    return text
