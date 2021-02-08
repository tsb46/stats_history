import re
from spacy.util import filter_spans


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


def replace_unwanted_characters(text):
    unwanted_strings = ["'", "’", "/"]
    for char in unwanted_strings:
        text = text.replace(char, '')
    return text
