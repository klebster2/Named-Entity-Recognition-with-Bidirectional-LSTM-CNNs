import pandas as pd
from pathlib import Path

class CorpusReader(object):

    def __init__(self, corpus_path:Path):
        if corpus_path.exists():
            self.corpus_path = corpus_path
        else:
            raise IOError(f'{corpus_path} does not exist.')
        self.df = self.read_data(corpus_path)

    @staticmethod
    def read_data(corpus_path):
        sentence_idx = 0
        df = pd.read_csv(
                corpus_path,
                sep=' ',
                names=["Word", "POS", "SynTag", "EntTag"],
                skip_blank_lines=False,
                quoting=3 # ignore quotes, read e.g. "
        )
        return df

    def add_sent_idx_conll03(self, return_bos:bool, return_eos:bool):
        """
        adds a sentence index to the dataframe

        :param return_bos:
        returns a boolean value telling the function to retain the boolean feature
        'sentence start' avoiding recomputation
        :param return_eos:
        returns a boolean value telling the function to retain the boolean feature
        'sentence end'

        optionally, we can return sentence starts and / or ends

        :return: None
        """
        # conll03: null rows mark the beginnings of sentences... mark them
        sent_spacer = self.df.isnull().values.all(1)

        sent_starts=(self.df.iloc[sent_spacer].index+1)[:-1]
        self.df.at[sent_starts, 'BOS'] = True
        self.df["BOS"].fillna(False, inplace=True)

        # add cumsum as sent idx to starts only
        self.df["sent_idx"] = self.df["BOS"].cumsum().fillna(method='ffill')

        if not return_bos:
            self.df.drop("BOS", axis=1, inplace=True)

        if return_eos:
            # Use similar logic to find the sentence ends
            sent_ends=(self.df.iloc[sent_spacer].index-1)[1:]
            self.df.at[sent_ends, 'EOS'] =  True
            self.df["EOS"].fillna(False, inplace=True)

        #self.df.dropna(inplace=True)

        # Cleanup:
        # the first row doesn't represent any useful information anymore...
        if self.df.iloc[0,:].isnull().values.all(0):
            self.df = self.df.iloc[1:,:]
        # also remove last null row if exists
        if self.df.iloc[-1,:].isnull().values.all(0):
            self.df = self.df.iloc[:-1,:] 

        # set to int
        self.df['sent_idx'] = self.df['sent_idx'].astype('int16')

        # give index name as we will have two indexes
        self.df.reset_index(inplace=True)
        self.df.index.rename('token', inplace=True)

        # now add sent idx
        self.df.set_index('sent_idx', append=True, inplace=True)


if __name__ == "__main__":
    # test
    
    import sys
    corpus_path=Path(sys.argv[1])

    corpus_df = CorpusReader(corpus_path)

    # add sent idx to dataframe given that we know that it is conll03 format
    corpus_df.add_sent_idx_conll03(True, True)

    df = corpus_df.df.copy()


