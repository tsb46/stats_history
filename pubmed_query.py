from Bio import Entrez


def get_summary_by_pubmed_id(pubmed_id_list, entrez_email):
    """
    :param pubmed_id_list: list of pubmed ids
    :return: list of dicts containing pubmed ids, abstract, date, etc.
    """
    Entrez.email = entrez_email
    n_ids = len(pubmed_id_list)
    article_dicts = {}
    # Esummary allows 10k articles per request
    # Loop through pubmed list until all ids have been captured
    for x in range(0, n_ids, 10000):
        print('# of articles fetched: {}'.format(len(article_dicts)))
        pubmed_id_subset = pubmed_id_list[x:x+10000]
        records_parsed = 0
        while records_parsed == 0:
            try:
                handle = Entrez.esummary(db='pubmed', id=','.join(pubmed_id_subset),
                                       rettype="xml", retmode="text")
                records = Entrez.read(handle, validate=False)
                handle.close()
                records_parsed = 1
                for record in records:
                    article_dicts.update(package_pubmed_article(record))
            except RuntimeError:
                print('parse failure... trying again')
                parsed_records = parse_failed_records(records)
                handle.close()
                article_dicts.update(parsed_records)
                records_parsed = 1

    return article_dicts


def parse_failed_records(records):
    article_dicts = {}
    for record in records:
        pmid = int(str(record['Id']))
        try:
            journal = record['FullJournalName']
            pub_date = record['PubDate']
            pmc_ref_count = record['PmcRefCount']
            article_info = {
                'journal': journal,
                'citations': pmc_ref_count,
                'pub_date': pub_date
            }
        except KeyError:
            article_info = None

        article_dict = {
            pmid: article_info
        }
        article_dicts.update(article_dict)
    return article_dicts


def package_pubmed_article(record):
    pmid = int(str(record['Id']))
    try:
        journal = record['FullJournalName']
        pmc_ref_count = record['PmcRefCount']
        pub_date = record['PubDate']
        article_info = {
            'journal': journal,
            'citations': pmc_ref_count,
            'pub_date': pub_date
        }
    except KeyError:
        print('key field does not exist')
        article_info = None

    article_dict = {
        pmid: article_info
    }
    return article_dict