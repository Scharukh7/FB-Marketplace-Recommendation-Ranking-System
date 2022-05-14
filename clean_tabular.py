#% %
from sqlalchemy import create_engine
import pandas as pd
import yaml
import numpy as np
import sklearn

class CleanTabularFB():
    
    def __init__(self):
        pass

    def connecting_to_RDS(self, creds: str='config/credentials.yaml'):
        with open(creds, 'r') as f:
            creds = yaml.safe_load(f)
        DATABASE_TYPE = 'postgresql'
        DBAPI = creds['DBAPI']
        ENDPOINT = creds['ENDPOINT']
        DBUSER = creds['DBUSER']
        DBPASSWORD = creds['DBPASSWORD']
        PORT = 5432
        DATABASE = creds['DATABASE']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{DBUSER}:"
                                    f"{DBPASSWORD}@{ENDPOINT}:"
                                    f"{PORT}/{DATABASE}")
        engine.connect()
        self.main_df = pd.read_sql_table(
            'products', self.engine,
            columns=["id", "product_name", "category", "product_description",
                     "price", "location",
                     "page_id", "create_time"])
        return self.main_df
               
    def outliers_remove_NA(self, column: str):
        temp_df = self.main_df[column].replace("N/A", np.nan)
        temp_df = temp_df.dropna()
        clean_df = pd.merge(temp_df, self.main_df, left_index=True, right_index=True)
        clean_df.drop(column + '_x', inplace=True, axis=1)
        clean_df.rename(columns={column +'_y':column}, inplace=True)
        self.main_df = clean_df

    def clean_price_sign(self):
        self.main_df['price'] = self.main_df['price'].str.strip('£')
        self.main_df['price'] = self.main_df['price'].str.replace(',','')
        self.main_df['price'] = df['price'].astype('float64')
        return self.main_df
    
    def remove_duplicates(df:pd):
        columns = [ "product_name", "product_description", "location"]
        df.drop_duplicates(subset=columns, keep="first", )

if __name__ == '__main__':
    clean = CleanTabularFB
    clean.connecting_to_RDS()
    clean.outliers_remove_NA()
    clean.clean_price_sign()
    clean.remove_duplicates()
   

