import pandas as pd

class Subsample:
    
    def __init__(self, pos_df, neg_df):
        self.pos_df = pos_df
        self.neg_df = neg_df
        
    def generate_percentages(self, num_intervals):
        total = 1.00
        inc_amo = 100 / num_intervals
        percentages = []
        for i in range(0, 100, inc_amo):
            percentages.append(i / 100)
        return percentages
        
    def generate_subsamples(self, percentiles):
        pos_num = self.pos_df.shape[0]
        neg_num = self.neg_df.shape[0]
        subsamples = {}
        for p in percentiles:
            pos_var = 'pos_' + str(int(p * 100))
            neg_var = 'neg_' + str(int(p * 100))
            subsamples[pos_var] = round(pos_num * p)
            subsamples[neg_var] = round(neg_num * p)
            pos_pt = 'pos_pt' + str(int(p * 100))
            neg_pt = 'neg_pt' + str(int(p * 100))
            subsamples[pos_pt] = self.pos_df.iloc[0:subsamples[pos_var], :]
            subsamples[neg_pt] = self.neg_df.iloc[0:subsamples[neg_var], :]
        return subsamples



