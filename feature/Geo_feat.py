# 1. Use Google API to get lat and lng
import numpy as np 
import pandas as pd 

print("Reading Data......")
test = pd.read_csv('~/.kaggle/competitions/avito-demand-prediction/test.csv', parse_dates=["activation_date"])
train = pd.read_csv('~/.kaggle/competitions/avito-demand-prediction/train.csv', parse_dates=["activation_date"])
print("Reading Done....")

import requests
lat = np.zeros(len(city_list))
long = np.zeros(len(city_list))
for index, item in enumerate(city_list):
    address = city_list[index]
    api_key = ""
    api_response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}'.format(address, api_key))
    api_response_dict = api_response.json()
    
    if api_response_dict['status'] == 'OK':
        lat[index] = api_response_dict['results'][0]['geometry']['location']['lat']
        long[index] = api_response_dict['results'][0]['geometry']['location']['lng']

location = pd.DataFrame({'city':city_list, 'lat':lat,'long':long})

# 2. Use Wikipedia to extract population
import os
from bs4 import BeautifulSoup
import requests

# Wikipedia
URL = 'https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Russia_by_population'

def scrape_wiki():
    wiki_dict = []
    
    r = requests.get(URL)

    # Soup
    bso = BeautifulSoup(r.text, 'lxml')
    
    # Table object
    tab = bso.body.find('table')
    
    # Rows
    rows = tab.find_all('tr')
    for row in rows[1:]:
        # Population
        city_name = row.find_all('td')[1].find_all('span')[0].text
    
        # Name (Russian)
        pop = int(row.find_all('td')[4].text.replace(',', ''))
        wiki_dict.append({'city': city_name, 'population': pop})
        print(city_name, pop)
    
    df = pd.DataFrame(data=wiki_dict)
    df = df.drop_duplicates('city')
    df.to_csv('city_population.csv')

# 3. Clustering by using HDBSCAN    
import hdbscan
location["lat_lon_hdbscan_cluster_05_03"] = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3
                                                                     ).fit_predict(location.loc[:, ["lat", "long"]])
location["lat_lon_hdbscan_cluster_10_03"] = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3
                                                                     ).fit_predict(location.loc[:, ["lat", "long"]])
location["lat_lon_hdbscan_cluster_20_03"] = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=3
location.to_csv('location.csv', index=False)                                                                   ).fit_predict(location.loc[:, ["lat", "long"]])

