'''
This script queries coin market cap, calculates close dates and makes predictions against the next day and stores the resutls as CSV.
'''

import requests
import csv
from bs4 import BeautifulSoup
import codecs
import datetime
from tqdm import tqdm
from _datetime import timedelta
import tensorflow as tf
from tensorflow.contrib import predictor
#from coinmarketcap_data_manip_v3 import get_all_data_calculations, get_calculated_headers, close_price_index, tag_index, get_two_week_history, _dummy_data
from dailycoins_scraper import write_coinmarket_historical, write_coinmarketcap_base_data
from dailycoins_generator import read_historic_coin_data, write_training_data, get_headers

print("**** Starting Coin PREDICTION...")

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def main():
    _none_value = "0"
    end_date = (datetime.datetime.now() + timedelta(days=-1)).strftime("%Y%m%d")
    start_date = (datetime.datetime.now() + timedelta(days=-15)).strftime("%Y%m%d")

    prediction_base_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\predictions\coinmarketcap_coins_base_for_prediction.csv"
    prediction_gen_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\predictions\coinmarketcap_data_" + start_date + "_to_" + end_date + "_for_predictions.csv"
    prediction_full_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\predictions\coinmarketcap_full_prediction_data.csv"
    
    all_coins_data_table = write_coinmarketcap_base_data(prediction_base_file, start_date, end_date)
    write_coinmarket_historical(prediction_gen_file, all_coins_data_table)
    coin_dict = read_historic_coin_data(prediction_gen_file)
    write_training_data(coin_dict, prediction_gen_file, prediction_full_file, training=False)
    predict_from_file(prediction_full_file)


def get_prediction_headers_from_file(prediction_file):
    header_list = []
    with codecs.open(prediction_file, 'r', "utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
        for line in tqdm(reader): 
            header_list = line
            break;
    return header_list

def predict_from_file(prediction_full_file):
    print("Starting predictions...")
    final_output_prediction_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\predictions\coinmarketcap_predictions.csv"
    model_dir = r'C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\saved_models\dailycoins'

    all_coins_market = []
    with codecs.open(prediction_full_file, 'r', "utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
        next(reader)
        for line in tqdm(reader):
            all_coins_market.append(line)

    serialized_list = []
    feature_list = []
    predict_fn = predictor.from_saved_model(model_dir)
    #predict_fn = None
    with open(final_output_prediction_file, "w") as csv_file:
        output_header_list = get_prediction_headers_from_file(prediction_full_file)
        output_header_list.append("predict_0")
        output_header_list.append("predict_1")
        csv_file.write("\t".join(output_header_list) + "\n")
        #text_file.write(str(serialized_proto_handle))
        for row in tqdm(all_coins_market):            
            feature_row = {} #For conversion to tensors
            # Convert the data to tensors
            for i in range(0, len(row)):
                if "name" != output_header_list[i] and "symbol" != output_header_list[i] and "tags" != output_header_list[i] and "markets" != output_header_list[i]:
                    feature_row[output_header_list[i]] = _float_feature(value=float(row[i]))
                else:
                    feature_row[output_header_list[i]] = _bytes_feature(value=str(row[i]).encode())
        
            example = tf.train.Example(features=tf.train.Features(feature=feature_row))
            serialized = example.SerializeToString()
            serialized_proto = tf.contrib.util.make_tensor_proto(serialized, shape=[1])
            serialized_proto_handle = serialized_proto.string_val
            predictions = predict_fn({"inputs" : serialized_proto_handle})
            row.append(str(predictions['scores'][0][0]))
            row.append(str(predictions['scores'][0][1]))
            csv_file.write("\t".join(str(x) for x in row) + "\n")



def odd():
    output_all_coins_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_coins_base.csv"
    _coin_prediction_output_file = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\predictions\coinmarketcap_coins_predictions_" + yesterday_date + ".csv"
    _coin_prediction_output_file_nopreds = r"C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\predictions\coinmarketcap_coins_predictions_" + yesterday_date + "no_preds.csv"
    model_dir = r'C:\Users\mmoxam\Desktop\Projects\Machine_Learning\data\saved_models\dailycoins'


    headarachi = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    requestor = requests.Session()

    # Cache all coins
    all_coins = []
    with codecs.open(output_all_coins_file, 'r', "utf-8") as csvfile:    
        reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
        next(reader)
        for line in tqdm(reader):
            path = line[2].split("?")[0]
            line[2] = path + "?start=" + week_ago_date  + "&end=" + yesterday_date
            all_coins.append(line)

    # Get all market data from yesterday for each coin
    all_coins_market = []
    for coin in tqdm(all_coins):
        #data_row = [coin[0], coin[1]]
        #data_row_two_days_ago = [coin[0], coin[1]]
        response = requestor.get(coin[3], headers=headarachi)
        soup_market = BeautifulSoup(response.text, 'html.parser')        

        # Get coin markets
        market_list = []
        table = soup_market.find("table", class_='table')
        if table:
            all_markets = table.find_all("tr")
            for i in range(1, len(all_markets)):
                tds = all_markets[i].find_all("td")
                market_list.append(tds[1].get_text().strip())
                if i >= 5:
                    break;

            response = requestor.get(coin[2], headers=headarachi)
            soup = BeautifulSoup(response.text, 'html.parser')

            if len(market_list) < 5:
                missing_mrkts = 5 - len(market_list)
                for j in range(0, missing_mrkts):
                    market_list.append("0")

            # Get coin tags
            label_list = []
            all_labels = soup.find_all("span", class_='label label-warning')
            for label in all_labels:
                label_list.append(label.get_text().strip())

            if len(label_list) < 1:
                label_list.append(_none_value)

            # Get historical price data
            days = []
            all_rows = soup.find_all('tr')
            del all_rows[0]
            if not "No data was found for the selected time period" in all_rows[0].text:
                for row in all_rows:
                    row_vals = row.find_all('td')
                    data_row = [coin[0], coin[1]]
                    for r in row_vals:
                        mod_val = r.get_text().strip().replace(",","")
                        mod_val = mod_val.replace("-", "0") if mod_val == "-" else mod_val
                        data_row.append(mod_val)
                        #data_row.append(r.get_text().strip().replace("-", "0").replace(",",""))
                    data_row.append(" ".join(label_list))
                    data_row = data_row + market_list
                    days.append(data_row)

                # Fix missing days if last 14 not found. Adding dummy rows with 0's
                num_days = len(days)
                if num_days < 14:
                    missing_days = 14 - num_days
                    for j in range(0, missing_days):
                        days.append(list(_dummy_data))

                    # Add better dummy day date including past date, name, symbol and tag
                    for k in range(num_days, 14):
                        prev_date = datetime.datetime.strptime(days[k - 1][2], '%b %d %Y')                            
                        dummy_date = (prev_date + timedelta(days=-1)).strftime("%b %d %Y")
                        days[k][0] = days[0][0]
                        days[k][1] = days[0][1]
                        days[k][2] = dummy_date
                        days[k][9] = days[0][9]

                high_low_spread = float(days[0][4]) - float(days[0][5])
                days[0].append(str(high_low_spread))

                high_low_spread_diff = float(days[1][4]) - float(days[1][5])
                days[0].append(str(high_low_spread - high_low_spread_diff))

                volume_diff = float(days[0][7]) - float(days[1][7])
                days[0].append(str(volume_diff))

                # Get 2 weeks of history
                week_data_row = get_two_week_history(days)
                for w_row in week_data_row:
                    days[0].append(str(w_row))
                        
                calcs = get_all_data_calculations(days)
                days[0].append(str(calcs["movement_one"]))
                days[0].append(str(calcs["movement_two"]))
                days[0].append(str(calcs["price_change_from_yesterday"]))
                days[0].append(str(calcs["percent_change_from_yesterday"]))
                days[0].append(str(calcs["price_change_from_two_days_ago"]))
                days[0].append(str(calcs["percent_change_from_two_days_ago"]))
                days[0].append(str(calcs["three_day_mavg"]))
                days[0].append(str(calcs["five_day_mavg"]))
                days[0].append(str(calcs["seven_day_mavg"]))    
                all_coins_market.append(days[0])
                print(days[0])



    with open(_coin_prediction_output_file_nopreds, "w") as csv_file:
        output_header_list = get_calculated_headers()    
        csv_file.write("\t".join(output_header_list) + "\n")
        for coin_row in tqdm(all_coins_market):
            csv_file.write("\t".join(str(x) for x in coin_row) + "\n")


    all_coins_market = []
    with codecs.open(_coin_prediction_output_file_nopreds, 'r', "utf-8") as csvfile:    
            reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
            next(reader)
            for line in tqdm(reader):            
                all_coins_market.append(line)



    serialized_list = []
    feature_list = []
    predict_fn = predictor.from_saved_model(model_dir)

    with open(_coin_prediction_output_file, "w") as csv_file:
        output_header_list = get_calculated_headers()
        output_header_list.append("predict_0")
        output_header_list.append("predict_1")
        csv_file.write("\t".join(output_header_list) + "\n")
        #text_file.write(str(serialized_proto_handle))
        for row in tqdm(all_coins_market):
            if not row[1] == "HAT":
                feature_row = {} #For conversion to tensors
                # Convert the data to tensors
                for i in range(0, len(row)):
                    #print(output_header_list[i] + " -> " + str(row[0]) + " -> " + str(row[i]))            
                    if i <= 2 or i == tag_index or "date" in output_header_list[i] or i == 10 or i == 11 or i== 12 or i == 13 or i == 14:
                        feature_row[output_header_list[i]] = _bytes_feature(value=str(row[i]).encode())
                    else:
                        feature_row[output_header_list[i]] = _float_feature(value=float(row[i]))
        
                example = tf.train.Example(features=tf.train.Features(feature=feature_row))
                serialized = example.SerializeToString()
                serialized_proto = tf.contrib.util.make_tensor_proto(serialized, shape=[1])
                serialized_proto_handle = serialized_proto.string_val
                predictions = predict_fn({"inputs" : serialized_proto_handle})
                row.append(str(predictions['scores'][0][0]))
                row.append(str(predictions['scores'][0][1]))        
                csv_file.write("\t".join(str(x) for x in row) + "\n")


    print("Complete...")

if __name__ == "__main__":
    main()
