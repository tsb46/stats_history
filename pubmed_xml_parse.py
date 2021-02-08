import pubmed_parser as pp
import tqdm

from multiprocessing import Pool

class PubmedParser:
    # Define class attribute - 'method_sec_markers'
    method_sec_markers = ['method', 'material', 'measure', 'analysis', 
    'statistical', 'experimental', 'analyses']

    def __init__(self, parallel_cores=1, verbose=False):
        self.verbose = verbose
        self.par_cores = parallel_cores

    def check_for_methods_section(self, pubmed_xml):
        """
        Check whether parsed pubmed xml output contains a 'methods' section
        :param pubmed_xml:  pubmed xml - list of dicts output from
        'parse_pubmed_paragrah'
        :return: boolean - 0 (false) or 1 (true)
        """
        section_list = [dict_info['section'] for dict_info in pubmed_xml
                        if dict_info['section'] != '']
        import pdb; pdb.set_trace()
        section_list = [dict_info['section'] for dict_info in pubmed_xml
                        if any(marker in dict_info['section'].lower()
                               for marker in self.method_sec_markers)
                        if dict_info['section'] != '']
        return len(section_list) > 0

    def build_pubmed_dict(self, metadata, body):
        """
        Build pubmed dictionary that contains all relevant
        information
        :param metadata: metadata xml parse list
        :param body: article body xml parse list
        :return: dictionary of relevant info for article
        """
        pubmed_dict = {
            'title': metadata['full_title'],
            'abstract': metadata['abstract'],
            'pmid': metadata['pmid'],
            'doi': metadata['doi'],
            'year': metadata['publication_year'],
            'date': metadata['publication_date'],
            'methods': self.pull_method_section(body)
        }
        return pubmed_dict

    def parse_all_xml(self, xml_folder):
        """
        Main runner function of class - returns a list of
        pubmed dicts from parsed xml files
        :param xml_folder: path to xml folder
        :return: list of pubmed dicts
        """
        xml_paths = pp.list_xml_path(xml_folder)
        # pool = Pool(self.par_cores)
        # if self.verbose:
        #     parsed_articles = list(tqdm.tqdm(pool.imap_unordered(
        #         self.parse_pubmed_article, xml_paths), total=len(xml_paths)))
        # else:
        #     parsed_articles = list(pool.imap_unordered(self.parse_pubmed_article,
        #                                     xml_paths))
        test = [self.parse_pubmed_article(path) for path in xml_paths]
        return parsed_articles

    def parse_pubmed_article(self, xml_path):
        """
        Parse pubmed xml file into human-readable format

        :param xml_path: path to xml file to parse
        :return: dictionary file containing relevant fields
        """
        body = pp.parse_pubmed_paragraph(xml_path)
        if self.check_for_methods_section(body):
            metadata = pp.parse_pubmed_xml(xml_path)
            pubmed_dict = self.build_pubmed_dict(metadata, body)
            return pubmed_dict
        else:
            return None

    def pull_method_section(self, body):
        """
        Pull method section text from parsed xml article
        :param body: parsed text of the 'body' of the article
        :return: string of full method section text
        """
        method_indx = [index for index, line in enumerate(body)
                       if any(marker in line['section'].lower()
                              for marker in self.method_sec_markers)
                       if line['section'] != '']
        method_text = ' '.join([line['text'] for index, line in enumerate(body)
                               if index in method_indx])
        return method_text












