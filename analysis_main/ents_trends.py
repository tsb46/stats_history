import numpy as np
import pandas as pd

from analysis_main.ents_base import EntityBase
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


class EntityTrends(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, precomp_ent_group=None):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts, precomp_ent_group)
        # Hard-coded metadata columns in the word count dataframe - intentionally 
        # dropped in some analyses
        self.metadata_cols = ['citations', 'journal', 
                              'title', 'abstract', 'domain']
    
    def count_by_domain_time(self, word_count_df, time_series, domains, 
                             start_year=2009, time_res='biannual', 
                             remove_most_recent=True):
        all_domains = []
        for domain in domains.unique():
            domain_mask = domains == domain
            domain_by_time = self.count_by_time(word_count_df.loc[domain_mask, :].copy(), 
                                                time_series, start_year, 
                                                time_res, remove_most_recent)
            domain_by_time['domain_tag'] = domain
            all_domains.append(domain_by_time.reset_index())
        return pd.concat(all_domains)
            
            
    def count_by_time(self, word_count_df, time_series, start_year=2009, 
                      time_res='biannual', remove_most_recent=True):
        word_count_df['date'] = pd.to_datetime(time_series, 
                                               format='%d-%m-%Y',
                                               errors='coerce')
        if time_res == 'quarterly':
            resampled_df = word_count_df.resample('Q', on='date').sum()
        elif time_res == 'biannual':
            resampled_df = word_count_df.resample('6M', on='date').sum()
        elif time_res == 'annual':
            resampled_df = word_count_df.resample('A', on='date').sum()
        else:
            raise Exception('Provided time resolution not available')
        resampled_df = self._threshold_dates(resampled_df, start_year, 
                                             remove_most_recent)
        # Drop 'article_counts' columns
        resampled_df = resampled_df.drop(columns='article_counts', errors='ignore')

        return resampled_df
    
    def decompose_time_series(self, word_count_df, time_series, 
                              n_comps, start_year=2009, time_res='biannual', 
                              remove_most_recent=True, norm='avg', win_sz=4):
        processed_df = self._preprocess_all_ts(word_count_df, time_series, 
                                               norm, time_res, start_year, 
                                               remove_most_recent)
        processed_df = processed_df.rolling(win_sz, min_periods=1).mean()
        pca = PCA(n_components=n_comps)
        pca.fit(processed_df.values)
        pca.scores = pca.transform(processed_df.values)
        return pca
        
    def plot_stat_ts(self, word_count_df, query,
                     start_year = 2009,
                     time_res = 'annual',
                     remove_most_recent=True):
        """
        param query: either a string or a list of strings referring 
        to the statistic entity(s) to track over time
        result: plot of timeseries        
        """
        if isinstance(query, str):
            query = [query]
        try:
            word_count_df['date'] = pd.to_datetime(word_count_df['date'], 
                                                   format='%d-%m-%Y',
                                                  errors='coerce')
        except KeyError:
            raise Exception('Word count df must include "date" column')
            
        query_df = word_count_df[query + ['date']].copy()
        resampled_df = self._resample_dates(query_df, time_res)
        resampled_df = self._threshold_dates(resampled_df, start_year, remove_most_recent)
        resampled_df.plot(figsize=(10,5))
        
        
    def plot_stat_ts_norm(self, word_count_df, query,
                          norm='avg',
                          start_year = 2009,
                          time_res = 'biannual', 
                          remove_most_recent=True):
        """
        param query: either a string or a list of strings referring 
        to the statistic entity(s) to track over time
        result: plot of timeseries        
        """
        if isinstance(query, str):
            query = [query]
        try:
            word_count_df['date'] = pd.to_datetime(word_count_df['date'], 
                                                   format='%d-%m-%Y',
                                                   errors='coerce')
        except KeyError:
            raise Exception('Word count df must include "date" column')
        
        query_df = word_count_df[query + ['date', 'article_counts']].copy()
        resampled_df = self._resample_dates(query_df, time_res)
        resampled_df = self._threshold_dates(resampled_df, start_year, remove_most_recent)
        if norm == 'perc':
            resampled_df = resampled_df[query].apply(
                lambda g: g / resampled_df['article_counts'], 
                axis=0)
        elif norm == 'avg':
            for ent in query:
                reg_count = LinearRegression().fit(
                    resampled_df['article_counts'].values.reshape(-1,1),
                    resampled_df[ent].values.reshape(-1,1)
                )
                resampled_df[ent] = (
                    resampled_df[ent].values.reshape(-1,1) -reg_count.predict(
                        resampled_df['article_counts'].values.reshape(-1,1)
                    )
                )
        resampled_df[query].plot(figsize=(10,5))
    
    @staticmethod
    def _resample_dates(df, time_res):
        if time_res == 'quarterly':
            resampled_df = df.resample('Q', on='date').sum()
        elif time_res == 'biannual':
            resampled_df = df.resample('6M', on='date').sum()
        elif time_res == 'annual':
            resampled_df = df.resample('A', on='date').sum()
        else:
            raise Exception('Provided time resolution not available')
        return resampled_df
    
    @staticmethod
    def _threshold_dates(df, start_year, remove_most_recent=False):
        time_mask = df.index.to_series() > pd.to_datetime(str(start_year))
        thres_df = df.loc[time_mask.values]
        if remove_most_recent:
            thres_df = thres_df.iloc[:-1].copy()
        return thres_df
    
    def _preprocess_all_ts(self, word_count_df, time_series,
                           norm, 
                           time_res, start_year,
                          remove_most_recent):
        word_count_df['date'] = pd.to_datetime(time_series, 
                                               format='%d-%m-%Y',
                                               errors='coerce')
        resampled_df = self._resample_dates(word_counts, time_res)
        resampled_df = self._threshold_dates(resampled_df, start_year, remove_most_recent)
        if norm == 'perc':
            resampled_df = resampled_df.apply(
                lambda g: g / resampled_df['article_counts'], 
                axis=0)
        elif norm == 'avg':
            for ent in resampled_df.columns:
                reg_count = LinearRegression().fit(
                    resampled_df['article_counts'].values.reshape(-1,1),
                    resampled_df[ent].values.reshape(-1,1)
                )
                resampled_df[ent] = (
                    resampled_df[ent].values.reshape(-1,1) -reg_count.predict(
                        resampled_df['article_counts'].values.reshape(-1,1)
                    )
                )
        resampled_df.drop(columns=['article_counts'], inplace=True)
        return resampled_df
    