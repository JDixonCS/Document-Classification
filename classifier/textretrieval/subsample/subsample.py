import pandas as pd


class Subsample:
    
    def __init__(self, pos_df, neg_df, dfs):
        self.pos_df = pos_df
        self.neg_df = neg_df
        self.dfs = dfs
        self.df_splits = []
        self.df_sens = []
        self.split_dataframes()

        
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

    def concat_and_print(self, pos_pt, neg_pt, num):
        df = pd.concat([pos_pt, neg_pt])
        print(f"DF{num}: {round(df.shape[0])}")

    def set_split_labels(self, inc_amo):
        labels = {}
        for i in range(1, inc_amo):
            labels[f"df{i}"] = f"Split {i}"
        return labels
    
    def set_iteration_labels(self, inc_amo):
        labels = {}
        for i in range(1, inc_amo):
            labels[f"it{i}"] = str(i)
        return labels

    def split_dataframes(self):
        for df in self.dfs:
            df_split = round(df.shape[0] * .5)
            self.df_splits.append(df_split)
            print(df_split)
            df_sen = round(df.shape[0])
            self.df_sens.append(df_sen)

    def print_df_splits(self):
        for split in self.df_splits:
            print(split)

    def print_df_sens(self):
        for sen in self.df_sens:
            print(sen)

    def get_column_data(df, start_idx, end_idx):
        return df.iloc[start_idx:end_idx, 5], df.iloc[start_idx:end_idx, 1]

    def print_shape(self, x, y):
        for i in range(1, len(x)+1):
            print(f"X{i}: {x[f'x{i}'].shape}")
        for i in range(1, len(y)+1):
            print(f"Y{i}: {y[f'y{i}'].shape}")