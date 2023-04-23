import os
import pandas as pd
import re

class ManuaLText:
    def __init__(self, negdata, posdata):
        self.negdata = negdata
        self.posdata = posdata

    def load_data(self):
        neg_lines = self._read_lines(self.negdata)
        pos_lines = self._read_lines(self.posdata)

        neg_df = self._create_dataframe(neg_lines)
        neg_df['label'] = 0

        pos_df = self._create_dataframe(pos_lines)
        pos_df['label'] = 1

        data_df = pd.concat([neg_df, pos_df], ignore_index=True)

        return data_df

    def _read_lines(self, file_path):
        with open(file_path, encoding="ISO-8859-1") as f:
            lines = f.readlines()
            f.close()

        # remove \n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()

        return lines

    def _create_dataframe(self, lines):
        data = {'sentence': []}
        i = 0

        for line in lines:
            first_col = re.sub(r' \(.*', '', line)
            data['sentence'].append(first_col)
            i += 1

        df = pd.DataFrame(data)

        return df
