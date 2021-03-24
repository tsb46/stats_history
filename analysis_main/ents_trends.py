import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from analysis_main.ents_base import EntityBase
from datetime import datetime
from sklearn.preprocessing import StandardScaler

np.random.seed(0) 

class EntityTrends(EntityBase):
    def __init__(self, ents_fp, ent_grouper=None, year_range=None, 
                 ignore_article_counts=True):
        super().__init__(ents_fp, ent_grouper, year_range, ignore_article_counts)
    
    def entity_count_by_domain_time(self, ents, time_series, domains, norm=True):
        resampled_df = ents.groupby([time_series, domains]).sum()
        if norm:
            article_counts = self._article_count(ents, time_series)
            resampled_df = resampled_df.apply(lambda x: x/article_counts, axis=0)
        return resampled_df
                
    def entity_count_by_time(self, ents, time_series, norm=True):
        resampled_df = ents.groupby(time_series).sum()
        if norm:
            article_counts = self._article_count(ents, time_series)
            resampled_df = resampled_df.apply(lambda x: x/article_counts, axis=0)
        return resampled_df
        
    def entity_ts(self, ents, time_series, domains=None, norm=True):
        """
        param query: either a string or a list of strings referring 
        to the statistic entity(s) to track over time
        result: plot of timeseries        
        """
        if norm:
            if domains is not None:
                article_counts = self._article_count(ents, time_series, domains)
                resampled_df = ents.groupby([time_series, domains]).sum()
            else:
                article_counts = self._article_count(ents, time_series)
                resampled_df = ents.groupby(time_series).sum()
            resampled_df = resampled_df.apply(lambda x: x/article_counts, axis=0)
        else:
            if domains is not None:
                resampled_df = query_df.groupby([time_series, domains]).sum()
            else:
                resampled_df = ents.groupby(time_series).sum()
        return resampled_df
    
    def entity_ts_bootstrap(self, ents, time_series, domains=None, n_bootstraps=100, norm=True):
        """
        result: plot of timeseries w/ bootstrapped standard errors      
        """
        if domains is not None:
            grouped_query = ents.groupby([time_series, domains])
            article_counts = self._article_count(ents, time_series, domains)
        else:
            grouped_query = ents.groupby(time_series)
            article_counts = self._article_count(ents, time_series)
        bootstrap_results=[]
        if norm:
            for n in range(n_bootstraps):
                resampled_series = grouped_query.apply(lambda x: x.sample(frac=1, replace=True).sum())
                resampled_series = resampled_series.apply(lambda x: x/article_counts, axis=0)
                bootstrap_results.append(resampled_series)
        else:
            for n in range(n_bootstraps):
                resampled_series = grouped_query.apply(lambda x: x.sample(frac=1, replace=True).sum())
                bootstrap_results.append(resampled_series)
        return pd.concat(bootstrap_results, axis=1)

    @staticmethod
    def trend_model(df, group_var, formula):
        """
        Trend modeling with generalized estimaing equations, accounting for dependency 
        structure (nesting) within journal
        """
        gee_fit = smf.gee(formula, group_var, data=df, family=sm.families.Binomial()).fit()
        return gee_fit

    @staticmethod
    def _article_count(word_count_df, time_series, domains=None):
        if domains is None:
            article_count = word_count_df.groupby(time_series).size()
        else:
            article_count = word_count_df.groupby([time_series, domains]).size()
        return article_count
    
    @staticmethod
    def _convert_year_to_int(year_series):
        year_series_int = pd.to_numeric(year_series, error='coerce', 
                                    downcast='integer')
        return year_series_int
    
    @staticmethod
    def _generate_samples_w_replacement(n_perm, entity_vec):
        n_samples = []
        for n_iter in range(n_perm):
            n_samples.append(np.random.choice(entity_vec, len(entity_vec)))
        return n_samples
    


    