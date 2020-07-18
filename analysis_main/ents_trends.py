import numpy as np
import pandas as pd

from analysis_main.ents_base import EntityBase
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class EntityTrends(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, precomp_ent_group=None):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts, precomp_ent_group)
        # Hard-coded metadata columns in the word count dataframe - intentionally 
        # dropped in some analyses
        self.metadata_cols = ['citations', 'journal', 
                              'title', 'abstract', 'domain']
    
    def entity_count_by_domain_time(self, word_count_df, time_series, domains):
        word_count_df['year'] = self._convert_pubmed_date_to_year(time_series)
        word_count_df['domain'] = domains
        resampled_df = word_count_df.groupby(['year', 'domain']).sum()
        return resampled_df
                
    def entity_count_by_time(self, word_count_df, time_series):
        word_count_df['year'] = self._convert_pubmed_date_to_year(time_series)
        resampled_df = word_count_df.groupby('year').sum()
        return resampled_df
        
    def entity_ts(self, word_count_df, time_series, query, 
                       norm=True, start_year = 2009):
        """
        param query: either a string or a list of strings referring 
        to the statistic entity(s) to track over time
        result: plot of timeseries        
        """
        if isinstance(query, str):
            query = [query]
        word_count_df['year'] = self._convert_pubmed_date_to_year(time_series)
        word_count_df = self._threshold_dates(word_count_df, start_year)
        query_df = word_count_df[query+['year']].copy()
        if norm:
            article_counts = self._article_count(query_df)
            resampled_df = query_df.groupby('year').sum()
            resampled_df = resampled_df.apply(lambda x: x/article_counts, axis=0)
        else:
            resampled_df = query_df.groupby('year').sum()
        return resampled_df
    
    def entity_ts_bootstrap(self, word_count_df, time_series, query, n_bootstraps=100,
                                 norm=True, start_year = 2009):
        """
        param query: a string referring 
        to the statistic entity to track over time
        result: plot of timeseries w/ bootstrapped standard errors      
        """

        word_count_df['year'] = self._convert_pubmed_date_to_year(time_series)
        word_count_df = self._threshold_dates(word_count_df, start_year)
        query_df = word_count_df[[query, 'year']].copy()
        grouped_query = query_df.groupby('year')[query]
        bootstrap_results=[]
        if norm:
            article_counts = self._article_count(query_df)
            for n in range(n_bootstraps):
                resampled_series = grouped_query.apply(lambda x: x.sample(frac=1, replace=True).sum())
                resampled_series/=article_counts
                bootstrap_results.append(resampled_series)
                
        else:
            for n in range(n_bootstraps):
                resampled_series = grouped_df.apply(lambda x: x.sample(frac=1, replace=True)).groupby(level=0).sum()
                bootstrap_results.append(resampled_series)
        return pd.concat(bootstrap_results, axis=1)
    
    @staticmethod
    def _article_count(word_count_df):
        return word_count_df.groupby('year').size()
    
    @staticmethod
    def _convert_pubmed_date_to_year(date_series):
        date_series = pd.to_datetime(date_series, 
                                   format='%d-%m-%Y',
                                   errors='coerce')
        return date_series.dt.year
    
    @staticmethod
    def _generate_samples_w_replacement(n_perm, entity_vec):
        n_samples = []
        for n_iter in range(n_perm):
            n_samples.append(np.random.choice(entity_vec, len(entity_vec)))
        return n_samples
    
    @staticmethod
    def _threshold_dates(word_count_df, start_year):
        return word_count_df.loc[word_count_df.year >= start_year, :].copy()

    