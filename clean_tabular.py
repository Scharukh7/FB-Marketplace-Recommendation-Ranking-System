#% %
from sqlalchemy import create_engine
import pandas as pd
import yaml
import numpy as np
import os
from os.path import isfile
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

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
        self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{DBUSER}:"
                                    f"{DBPASSWORD}@{ENDPOINT}:"
                                    f"{PORT}/{DATABASE}")

    def download_json(self, table: str):
        #connects to RDS database and downloads the data and converts to json file and saves inside the data folder
        file_path = "data/" + table + "_table.json"
        if isfile(file_path) == True:
            print(file_path, "is already download, skipping")
        else:
            print("connecting to DB..")
            engine = self.engine
            main_df = pd.read_sql_table(table, engine)
            print("Saved as ", file_path)
            main_df.to_json(file_path)
    
    def read_json(self):
        json_file = "data/products_table"
        if isfile(json_file) == True:
            print("Getting json file..")
            self.fb_df = pd.read_json(json_file)
            print("json file acquire", self.fb_df)
        else:
            print("json file already acquired")
         
    def outliers_remove_NA(self):
        column: str
        #clean the data which includes missing values into panda frame np.nan
        temp_df = self.fb_df[column].replace("N/A", np.nan)
        temp_df = temp_df.dropna()
        #merge the new data
        clean_df = pd.merge(temp_df, self.fb_df, left_index=True, right_index=True)
        #drop duplicated columns
        clean_df.drop(column + '_x', inplace=True, axis=1)
        #rename the columns
        clean_df.rename(columns={column +'_y':column}, inplace=True)
        self.fb_df = clean_df
        return self.fb_df

    def clean_price_sign(self):
        #remove the '£' sign from the price
        self.fb_df['price'] = self.fb_df['price'].apply(
            lambda x: x.strip("£").replace(',',''))
        #change the datatype of new price from object to float64
        self.fb_df['price'] = self.fb_df['price'].astype('float64')

    def remove_outliers_from_price(self):
        #remove the outliers formed from the max and min price on the plot fig
        self.fb_df = self.fb_df[self.fb_df['price'] < 1000]
        self.fb_df = self.fb_df[self.fb_df['price'] > 1]
        return self.fb_df 
    
    def split_category_column(self):
        #split the category column into two, first 'main_category'
        self.fb_df["main_category"] = self.fb_df["category"].apply(
            lambda x: x.split("/")[0].strip())
        #then split into 'sub_category'
        self.fb_df["sub_category"] = self.fb_df["category"].apply(
            lambda x: x.split("/")[1].strip())
    
    def geocode_columns_from_locations(self):
        #iterate through location column and get the latitude and longitude from cities
        geocoder = Nominatim(user_agent='GetLoc')
        geocode = RateLimiter(geocoder.geocode, min_delay_seconds=0.3, return_value_on_exception=None)
        #location = geocode.reverse(, timeout=10, language='en')
        self.fb_df['geo_location'] = self.fb_df['location'].apply(geocode)
        self.fb_df['longiude_latitude'] = self.fb_df['geo_location'].apply(
        lambda loc: tuple(loc.point) if loc else None)
    
    def convert_column_to_num(self, column: str):
        #convert the text data into number for the ML model
        #get rid of any special characters from the columns
        self.fb_df[column] = self.fb_df[column].str.lower().replace('[^0-9a-zA-Z]+', '_', regex=True)
        #category encodings
        category_encodings = pd.get_dummies(
            self.fb_df[column], prefix=column, drop_first=True)
        #merge the cleaned dataframes
        self.fb_df = pd.concat(
            [self.fb_df, category_encodings], axis=1)

    def remove_duplicate_data(self):
        print("Removing duplicates.")
        columns = ["product_name", "product_description", "location"]
        self.fb_df.drop_duplicates(subset=columns, keep="first", )
    
    def save_cleaned_data(self):
        save_path = "data/cleaned_tabular_data.json"
        clean_data = self.fb_df
        clean_data = clean_data.to_json(save_path)
        print("cleaned tabular data saved in", save_path)
   
if __name__ == '__main__':
    clean = CleanTabularFB()
    clean.connecting_to_RDS()
    clean.download_json('products')
    clean.read_json()
    clean.outliers_remove_NA()
    clean.clean_price_sign()
    clean.remove_outliers_from_price()
    clean.split_category_column()
    clean.geocode_columns_from_locations()
    clean.convert_column_to_num('main_category')
    clean.save_cleaned_data()

   

