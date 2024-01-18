# -*- coding: UTF-8 -*-
import pandas as pd
import model.config as config

# 這裡是讀取 紀錄分類的csv,xlsx檔案
cat_ratios_path = str(config.path / "thesis_code/category/ratios cat.xlsx")
cat_ch204_path = str(config.path / "thesis_code/category/list and cat of 204 firm variables.xlsx")
cat_macro_path = str(config.path / "thesis_code/category/macro cat.xlsx")
macor_data_path = str(config.path / "thesis_code/portfolios_data/macro.csv")


class DataLoader:
    def __init__(self, start='1980-07-01', end='2021-12-01'):
        self.start, self.end = start, end

    def get_macro(self, df):

        self.macro = pd.read_csv(macor_data_path)
        self.macro.rename(columns={'sasdate': 'date'}, inplace=True)
        self.macro = self.macro.set_index(pd.to_datetime(self.macro['date'], format='%m/%d/%Y')).drop(['date'], axis=1)
        df = df.merge(self.macro, on=['date'])

        return df

    def get_categories(self, df):
        self.cat_ratios = pd.read_excel(cat_ratios_path)
        self.cat_ratios = self.cat_ratios[['Variable Name', 'cat for thesis']]
        self.cat_ch204 = pd.read_excel(cat_ch204_path)
        self.cat_ch204 = self.cat_ch204[['Variable Name', 'cat for thesis']]
        self.cat_macro = pd.read_excel(cat_macro_path)
        self.cat_macro = self.cat_macro[['Variable Name', 'cat for thesis']].astype(str)

        # 分構面: 之後讀csv整理的檔案，放入串列中，不要手動填入。
        self.sector_l = {
            'f': [],  # financial
            'o': [],  # operating
            'p': [],  # technical
            'sup': [],  # supply
            'sen': [],  # sentiment
            'esg': [],  # ESG
            # --------------------------------------Macro Econmoic-----之後改名稱，1:Output and income...----------------------------------------------- #
            '1': [],  # Group 1
            '2': [],  # Group 2
            '3': [],  # Group 3
            '4': [],  # Group 4
            '5': [],  # Group 5
            '6': [],  # Group 6
            '7': [],  # Group 7
            '8': [],  # Group 8
            '9': []  #
        }

        for i in df.columns:
            if ((i in list(self.cat_ch204['Variable Name'])) or (i[:-3] in list(self.cat_ch204['Variable Name']))):
                for n in range(len(self.cat_ch204)):
                    if ((i == self.cat_ch204['Variable Name'][n]) or (
                            i[:-3] == self.cat_ch204['Variable Name'][n])) and i not in self.sector_l[
                        self.cat_ch204['cat for thesis'][n]]:
                        self.sector_l[self.cat_ch204['cat for thesis'][n]].append(i)

            if (i in list(self.cat_ratios['Variable Name'])) or (i[:-3] in list(self.cat_ratios['Variable Name'])):
                for n in range(len(self.cat_ratios)):
                    if ((i == self.cat_ratios['Variable Name'][n]) or (
                            i[:-3] == self.cat_ratios['Variable Name'][n])) and i not in self.sector_l[
                        self.cat_ratios['cat for thesis'][n]]:
                        self.sector_l[self.cat_ratios['cat for thesis'][n]].append(i)

            if (i in list(self.cat_macro['Variable Name'])) or (i[:-3] in list(self.cat_macro['Variable Name'])):
                for n in range(len(self.cat_macro)):
                    if ((i == self.cat_macro['Variable Name'][n]) or (
                            i[:-3] == self.cat_macro['Variable Name'][n])) and i not in self.sector_l[
                        self.cat_macro['cat for thesis'][n]]:
                        self.sector_l[self.cat_macro['cat for thesis'][n]].append(i)
        return self.sector_l
