import requests as r
from pandas.io.json import json_normalize
import json
# from ph2 import ParseHub
# from bs4 import BeautifulSoup

### An easy method to retrieve data from live charts ###
# https://medium.com/analytics-vidhya/an-easy-technique-for-web-scraping-an-interactive-web-chart-38f5f945ca63

## Index Values All Art Index Family ##
def getindexvalues_allartindexfamily():
    # url to get
    url1 = 'https://www.artmarketresearch.com/indexes/getallartdual.php?sd=197801&ed=202310&alpha=0.2&numartists=10000&minsales=1&vorf=f'
    res = r.get(url1)
    search_cookies = res.cookies

    # post method data
    get_data = {'method':'GET', "sd":"197801","ed":"202310", "alpha":"0.2", 
                "numartists":"10000", "minsales":"1", "vorf":"f"}

    # headers information
    headers = {'user-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}

    # request get data
    res_get = r.post(url1, data=get_data , cookies=search_cookies, headers = headers)

    # pull data in json format
    index_values = res_get.json()
    index_headers = res_get.json()["cols"]
    df = json_normalize(index_headers)

    return index_values

## Stability Values All Art Index Family ##
def getstabilityvalues_allartindexfamily():
    url2 = 'https://www.artmarketresearch.com/indexes/getallartchanges.php?sd=197801&ed=202310&minsales=1&numartists=10000'
    res = r.get(url2)
    search_cookies = res.cookies

    get_data = {'method':'GET', "sd":"197801", "ed":"202310", 
                "minsales":"1", "numartists":"10000"}
    
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'}

    res_get = r.post(url2, data=get_data, cookies=search_cookies, headers=headers)

    stability_values = res_get.json()
    index_headers = res_get.json()["cols"]
    df = json_normalize(index_headers)

    return stability_values

## Price Values GBP All Art Index Family ##
def getpricevalues_allartindexfamily():
    url3 = 'https://www.artmarketresearch.com/indexes/getallartdual.php?sd=197801&ed=202310&alpha=0.2&numartists=10000&minsales=1&vorf=v'
    res = r.get(url3)
    search_cookies = res.cookies

    get_data = {'method':'GET', "sd":"197801", "ed":"202310", "alpha":"0.2", 
                "numartists":"10000", "minsales":"1", "vorf":"v"}
    
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'}

    res_get = r.post(url3, data=get_data, cookies=search_cookies, headers=headers)

    price_values = res_get.json()
    index_headers = res_get.json()["cols"]
    df = json_normalize(index_headers)

    return price_values

def getstabilityvalues_allartindex():
    url4 = 'https://www.artmarketresearch.com/indexes/getallartchanges.php?sd=200701&ed=202310&minsales=1&numartists=10000'
    res = r.get(url4)
    search_cookies = res.cookies

    get_data = {'method':'GET', "sd":"200701", "ed":"202310", 
                "minsales":"1", "numartists":"10000"}
    
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'}

    res_get = r.post(url4, data=get_data, cookies=search_cookies, headers=headers)

    stability_values = res_get.json()
    index_headers = res_get.json()["cols"]
    df = json_normalize(index_headers)

    return stability_values


### Execution ###
if __name__ == "__main__":

    # Save index_values all art index family data
    index_values = getindexvalues_allartindexfamily()
    index_values_flag = False # Only run once to save data
    if(index_values_flag):
        with open('price_values.json', 'w') as f:
            json.dump(index_values, f)
    
    # Save stability_values all art index family data
    stability_values = getstabilityvalues_allartindexfamily()
    stability_values_flag = False # Only run once to save data
    if(stability_values_flag):
        with open('stability_values.json', 'w') as f:
            json.dump(stability_values, f)

    # Save price_values (GBP) all art index family data
    price_values = getpricevalues_allartindexfamily()
    price_values_flag = False # Only run once to save data
    if(price_values_flag):
        with open('price_values.json', 'w') as f:
            json.dump(price_values, f)

    # Save stability values all art index data (different from all art index family)
    stability_values_allartindex = getstabilityvalues_allartindex()
    stability_values_allartindex_flag = True # Only run once to save data
    if(stability_values_allartindex_flag):
        with open('stability_values.json', 'w') as f:
            json.dump(stability_values_allartindex, f)


### Experimental ###

## Parsehub for scraping ##

# API DOCS : https://www.parsehub.com/docs/ref/api/v2/#run-a-project

# List of projects
# params = {
#   "api_key": "tdEhTTxW-JCU",
#   "offset": "0",
#   "limit": "20",
#   "include_options": "1"
# }
# r = requests.get('https://www.parsehub.com/api/v2/projects', params=params)
# print(r.text)

# # Get a project
# params = {
#   "api_key": "tdEhTTxW-JCU",
#   "offset": "0",
#   "include_options": "1"
# }
# r = requests.get('https://www.parsehub.com/api/v2/projects/{PROJECT_TOKEN}', params=params)
# print(r.text)

# # Run a project
# params = {
#   "api_key": "tdEhTTxW-JCU",
#   "start_url": "http://www.example.com",
#   "start_template": "main_template",
#   "start_value_override": "{\"query\": \"San Francisco\"}",
#   "send_email": "1"
# }
# r = requests.post("https://www.parsehub.com/api/v2/projects/{PROJECT_TOKEN}/run", data=params)

# print(r.text)


