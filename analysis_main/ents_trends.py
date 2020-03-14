import numpy as np
import pandas as pd

from analysis_main.ents_base import EntityBase
from datetime import datetime


class EntityTrends(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, default_time_res='quarterly'):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts)
        self.default_time_res = default_time_res
    

    def plot_stat_ts(self, word_count_df, query,
                     start_year = 2008,
                     time_res = 'annual'):
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
            
        time_mask = word_count_df['date'] > pd.to_datetime(str(start_year))
        word_count_df = word_count_df.loc[time_mask, :]
        query_df = word_count_df[query + ['date']].copy()
        query_df = query_df.set_index('date')
        if time_res == 'quarterly':
            resampled_df = query_df.apply(self._resample_dates_quarterly, axis=0)
        elif time_res == 'biannual':
            resampled_df = query_df.apply(self._resample_dates_biannual, axis=0)
        elif time_res == 'annual':
            resampled_df = query_df.apply(self._resample_dates_annual, axis=0)
        else:
            raise Exception('Provided time resolution not available')
        resampled_df.plot(figsize=(10,5))
        
        
    def plot_stat_ts_perc(self, word_count_df, query,
                          start_year = 2008,
                          time_res = 'annual'):
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
            
        time_mask = query_df['date'] > pd.to_datetime(str(start_year))
        query_df = query_df.loc[time_mask]

        if time_res == 'quarterly':
            resampled_df = query_df.resample('Q', on='date').sum()
        elif time_res == 'biannual':
            resampled_df = query_df.resample('6M', on='date').sum()
        elif time_res == 'annual':
            resampled_df = query_df.resample('A', on='date').sum()
        else:
            raise Exception('Provided time resolution not available')
        resampled_df = resampled_df[query].apply(
            lambda g: g / resampled_df['article_counts'], 
            axis=0)
        resampled_df.plot(figsize=(10,5))
        
    @staticmethod
    def _resample_dates_annual(series, perc):
        resampled_series = series.resample('A').sum()
        return resampled_series
    
    @staticmethod
    def _resample_date_biannual(series, perc):
        resampled_series = series.resample('6M').sum()
        return resampled_series
    
    @staticmethod
    def _resample_dates_quarterly(series, perc):
        resampled_series = series.resample('Q').sum()
        return resampled_series
    