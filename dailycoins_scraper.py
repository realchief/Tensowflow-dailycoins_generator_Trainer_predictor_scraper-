import requests
import calendar
from bs4 import BeautifulSoup
import codecs
import datetime
from tqdm import tqdm
import traceback


def main():
    start_date = '20130428'
    end_date = '20180109'

    output_all_coins_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_coins_base.csv"
    output_training_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_data_" + start_date + "_to_" + end_date + ".csv"
    
    all_coins_data_table = write_coinmarketcap_base_data(output_all_coins_file, start_date, end_date)
    write_coinmarket_historical(output_training_file, all_coins_data_table)


def write_coinmarketcap_base_data(output_all_coins_file, start_date, end_date):
    ''' Get all coin names and historical urls from coin market cap and stores to a file '''    
    
    #request_url = r"https://coinmarketcap.com/currencies/raiblocks/historical-data/?start=" + start_date  + "&end=" + end_date
    
    headarachi = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    requestor = requests.Session()
    base_url = "https://coinmarketcap.com"
    all_coins_url = base_url + r"/all/views/all/"
    
    response = requestor.get(all_coins_url, headers=headarachi)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_coins_rows = soup.find_all('tr') 
    all_coins_data_table = []

    print("All coins: " + str(len(all_coins_rows)))
    for row in tqdm(all_coins_rows):
        row_vals = row.find_all('td')
        if len(row_vals) > 0:
            coin_a = row_vals[1].find("a", class_='currency-name-container')
            coin_name = coin_a['href'].split("/")[2].replace("-"," ").title()
            coin_symbol = row_vals[2].get_text().strip()
            coin_data_url = base_url + coin_a['href'] + "historical-data/?start=" + start_date  + "&end=" + end_date
            market_data_url = base_url + coin_a['href'] + "#markets/"
            all_coins_data_table.append((coin_name, coin_symbol, coin_data_url, market_data_url))
        else:
            print("Skipping row: " + str(row))

    with codecs.open(output_all_coins_file, 'w', "utf-8") as csvfile:    
        header_list = ["name", "symbol", "url", "url_markets"]
        csvfile.write("\t".join(header_list) + "\n")
        for coin in all_coins_data_table:
            csvfile.write("\t".join(coin) + "\n")

    return all_coins_data_table

def write_coinmarket_historical(output_training_file, all_coins_data_table):
    ''' Write historical and market data to new file '''
    _none_value = "0"    

    headarachi = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    requestor = requests.Session()

    #Get each coin data
    with codecs.open(output_training_file, 'w', "utf-8") as csvfile:
        header_list = ["name", "symbol", "date", "open", "high", "low", "close", "volume", "market_cap", "tags", "markets"]
        csvfile.write("\t".join(header_list) + "\n")

        for coin in tqdm(all_coins_data_table):
            response = requestor.get(coin[3], headers=headarachi)
            soup_market = BeautifulSoup(response.text, 'html.parser')
        
            # Get coin markets
            market_list = []
            try:
                all_markets = soup_market.find(id="markets-table").find_all("tr")        
                for i in range(1, len(all_markets)):
                    tds = all_markets[i].find_all("td")
                    mrkt = tds[1].get_text().strip()
                    if not mrkt in market_list:
                        market_list.append(mrkt)
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())

            if len(market_list) < 1:
                market_list.append("None")

            # Get historical coin data
            response = requestor.get(coin[2], headers=headarachi)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get coin tags
            label_list = []
            all_labels = soup.find_all("span", class_='label label-warning')
            for label in all_labels:
                label_list.append(label.get_text().strip())

            if len(label_list) < 1:
                label_list.append("None")

            all_rows = soup.find_all('tr')
            for row in all_rows:
                row_vals = row.find_all('td')
                if len(row_vals) > 2:
                    data_row = [coin[0], coin[1]]
                    for r in row_vals:
                        mod_val = r.get_text().strip().replace(",","")
                        mod_val = mod_val.replace("-", "0") if mod_val == "-" else mod_val
                        data_row.append(mod_val)
                    data_row.append(" ".join(label_list))
                    data_row.append(" ".join(market_list))
                    csvfile.write("\t".join(data_row) + "\n")
        
            
    print("Complete...")

if __name__ == "__main__":
    main()
