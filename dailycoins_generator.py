import csv
import codecs
from tqdm import tqdm
from datetime import datetime
from _datetime import timedelta
import re
import traceback
from _datetime import timedelta
import time

def get_headers(input_coinmarket_data, training=True):
    ''' Returns a list of all field headers that will be exported ''' 

    header_list = []
    with codecs.open(input_coinmarket_data, 'r', "utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
        for line in tqdm(reader): 
            header_list = line
            break;

    header_list.append("will_increase")
    header_list.append("total_market_volume")
    header_list.append("total_market_cap")
    header_list.append("high_low_spread")
    header_list.append("volume_difference")    
    header_list.append("three_day_mavg")
    header_list.append("five_day_mavg")
    header_list.append("seven_day_mavg")
    header_list.append("price_change_yesterday")
    header_list.append("percent_change_yesterday")
    header_list.append("price_change_two_days")
    header_list.append("percent_change_two_days")
    header_list.append("price_move_yesterday")
    header_list.append("price_move_two_days")
    

    truncated_header = list(header_list)
    del truncated_header[9] # Delete tags, market fields and will_increase for last 14 days data. FYI: Deletion alters the index count hence position remains the same
    del truncated_header[9]
    del truncated_header[9]
    truncated_header.append("will_increase_next_day")

    dynamic_header_list = []
    for i in range(1, 15):
        for j in range(2, len(truncated_header)):
            dynamic_header_list.append(truncated_header[j] + "_" + str(i))

    if not training:
        del header_list[11]  # removing will increase label field

    return header_list + dynamic_header_list

def get_price_change_from_yesterday(today, yesterday, close_index):
    yesterday_close_price = float(yesterday[close_index])
    price_change = float(today[close_index]) - yesterday_close_price
    if yesterday_close_price == 0:
        return [price_change, 0]
    else:
        percent_change = price_change / yesterday_close_price
        return [price_change, percent_change]

def get_price_change_from_two_days_ago(today, two_days_ago, close_index):    
    close_price_two = float(two_days_ago[close_index])
    price_change = float(today[close_index]) - close_price_two        
    if close_price_two == 0:
        return [price_change, 0]
    else:
        percent_change = price_change / close_price_two
        return [price_change, percent_change]   

def get_price_movement_yesterday(today, yesterday, close_index):
    today_close_price = float(today[close_index])
    previous_close_price = float(yesterday[close_index])
    value = 0
    if previous_close_price > today_close_price:
        value = 1
    elif previous_close_price < today_close_price:
        value = 2
    return value

def get_price_movement_two_days_ago(today, two_days_ago, close_index):
    today_close_price = float(today[close_index])
    previous_two_days_close_price = float(two_days_ago[close_index])
    value = 0
    if previous_two_days_close_price > today_close_price:
        value = 1
    elif previous_two_days_close_price < today_close_price:
        value = 2
    return value

def get_moving_avg(days, num_days_to_avg, line_start_index, close_index):
    ''' Calculates the specified amount of days moving average '''
    close_sum = 0.0
    counter = 0
    for i in range(0, num_days_to_avg):
        day_index = line_start_index + i
        if day_index < len(days):
            close_sum += float(days[day_index][close_index])
            counter += 1
    return close_sum / counter
    '''
    max_value = num_days_to_avg if len(days) >= num_days_to_avg else len(days)
    for i in range(0, max_value):
        day_count += float(days[line_start_index + i][close_index])
    
    return day_count / max_value
    '''

def get_two_week_history(two_weeks):
    ''' Get's historic data for last 2 weeks. If days are missing they are added in as dummy data but not only for last missing days '''
    #header_list = ["name", "symbol", "date", "open", "high", "low", "close", "volume", "market_cap", "coin"]    
    data_row = []
    spread_list = []
    num_days = len(two_weeks)

    for i in range(1, num_days):
        spread = float(two_weeks[i][4]) - float(two_weeks[i][5])        
        new_row = [two_weeks[i][2], two_weeks[i][3], two_weeks[i][4], two_weeks[i][5], two_weeks[i][6], two_weeks[i][7], two_weeks[i][8], spread]
        data_row = data_row + new_row       

    return data_row

def read_historic_coin_data(input_coinmarket_historic_data):

    # Cache coin data in sequence    
    print("Starting Coin caching...")
    coin_dict = {}
    coin_dict_date = {}
    date_vol_dict = {}
    date_mcap_dict = {}
    with codecs.open(input_coinmarket_historic_data, 'r', "utf-8") as csvfile:    
        reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
        next(reader)
        for line in tqdm(reader):
            symbol = line[0] + "-$" + line[1]
            
            #date = datetime.datetime.strptime(line[2], '%b %d %Y') #Parse Date
            if symbol in coin_dict:
                coin_dict[symbol].append(line)
            else:
                coin_dict[symbol] = [line]
            
            date = line[2]
            if date in coin_dict_date:
                coin_dict_date[date].append(line)
            else:
                coin_dict_date[date] = [line]
            
            #date = datetime.datetime.strptime(line[2], '%b %d %Y') #Parse Date            

        for k,v in coin_dict_date.items():
            cnt = 0
            for row in v:
                cnt += float(row[7])
            date_vol_dict[k] = cnt

        for l,m in coin_dict_date.items():
            cntm = 0
            for rowa in m:
                cntm += float(rowa[8])
            date_mcap_dict[l] = cntm

    return [coin_dict, date_vol_dict, date_mcap_dict]


def main():
    input_coinmarket_historic_data = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_data_20130428_to_20180109.csv"
    output_coinmarket_training_data = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_training_data.csv"

    coin_dict_list = read_historic_coin_data(input_coinmarket_historic_data)
    write_training_data(coin_dict_list, input_coinmarket_historic_data, output_coinmarket_training_data)


def get_will_increase(future_day, today, high_index, close_index):
    ''' Gets whether or not the high for the future is greater than 35% for the close of today '''

    will_increase = 0
    today_close_price = float(today[close_index])
    if today_close_price > 0:
        future_high_diff = float(future_day[high_index]) - today_close_price  # Difference between the high in the future vs the close of today
        percent_diff = future_high_diff / today_close_price
        if percent_diff > 0.35:
            will_increase = 1 # Specify the label as 1 if > than 35% next 2 days

    return will_increase

def get_high_low_spread(today, high_index, low_index):
    ''' Gets the difference between the high and the low for today'''
    return float(today[high_index]) - float(today[low_index]) # The difference between the high and low for today

def get_volume_difference(today, yesterday, volume_index):
    ''' Gets the difference between the volume for yesterday and today'''
    return float(today[volume_index]) - float(yesterday[volume_index]) # The difference between the high and low for today

def get_previous_day(line, current_index, num_of_days, _dummy_day, day_count):
    ''' Gets the day calculated as today - day count '''
    yesterday = list(_dummy_day) # Start with dummy data
    yesterday_index = current_index + day_count
    if yesterday_index < num_of_days:
        yesterday = list(line[yesterday_index])

    return yesterday


def write_training_data(coin_dict_list, input_coinmarket_historic_data, output_coinmarket_training_data, training=True):

    print("Starting Manipulations...")
    date_index = 2
    high_index = 4
    low_index = 5
    close_index = 6
    volume_index = 7
    _dummy_day = [0,0,0,0,0,0,0,0,0,0,0]
    _date_format = "%b %d %Y"    

    coin_dict = coin_dict_list[0]
    vol_dict = coin_dict_list[1]
    cap_dict = coin_dict_list[2]

    print("Length Coins: " + str(len(coin_dict)))
    print("Length Vol: " + str(len(coin_dict)))
    print("Length Cap: " + str(len(coin_dict)))
    
    # Write coin data with calculations to file
    with codecs.open(output_coinmarket_training_data, 'w', "utf-8") as out_file:
        header = get_headers(input_coinmarket_historic_data, training=training)
        #header_list = ["name", "symbol", "date", "open", "high", "low", "close", "volume", "market_cap", "coin", "24_movement", "24_close_price", "24_close_percent", "next_day_fifty_percent"]
        #header = get_calculated_headers(input_coinmarket_historic_data, include_tomorrow=True)
        out_file.write("\t".join(header) + "\n")
        
        start_index = 2 if training else 0
        for coin, line in tqdm(coin_dict.items()):

            try:
                # Adhoc fix for faulty data
                if coin == "$EMV": # Apparntly teh clsoe price has a 0 that throws off calculations. Data flaw
                    for row in line:
                        if row[0] == "Ethereum Movie Venture" and row[1] == "EMV" and row[2] == "May 28 2017":
                            row[close_index] = "0.418567"
                            row[close_index - 1] = "0.196942"

                num_of_days = len(line)
                for i in range(start_index, num_of_days):
                    today = list(line[i])
                    today_date_str = today[date_index]
                    today_date = datetime.strptime(today[date_index], _date_format)
                    today[date_index] = int(time.mktime(time.strptime(today[date_index], _date_format))) - time.timezone
                    yesterday = get_previous_day(line, i, num_of_days, _dummy_day, 1)
                    two_days_ago = get_previous_day(line, i, num_of_days, _dummy_day, 2)

                    if training:
                        future_day = line[i-2]
                        will_increase = get_will_increase(future_day, today, high_index, close_index)
                        today.append(will_increase)
                    
                    # Additional Fields                    
                    today.append(vol_dict[today_date_str]) # total volume 
                    today.append(cap_dict[today_date_str]) # total market cap
                    today.append(get_high_low_spread(today, high_index, low_index)) # high low spread
                    today.append(get_volume_difference(today, yesterday, volume_index)) # volume difference from yesterday
                    today.append(get_moving_avg(line, 3, i, close_index)) # 3 day moving average
                    today.append(get_moving_avg(line, 5, i, close_index)) # 5 day moving average
                    today.append(get_moving_avg(line, 7, i, close_index)) # 7 day moving average
                    change = get_price_change_from_yesterday(today, yesterday, close_index)
                    change_two = get_price_change_from_two_days_ago(today, two_days_ago, close_index)
                    today.append(change[0]) # price change from yesterday
                    today.append(change[1]) # percent change from yesterday
                    today.append(change_two[0]) # price change from 2 days ago
                    today.append(change_two[1]) # percent change from 2 days ago
                    today.append(get_price_movement_yesterday(today, yesterday, close_index)) # get price movement value from yesterday
                    today.append(get_price_movement_two_days_ago(today, two_days_ago, close_index)) # get price movement value from two days ago                    

                    for j in range(1, 15): # Grab last 14 days of data
                        row_to_add = list(_dummy_day) # Start with dummy data
                        row_to_add_index = i + j
                    
                        if row_to_add_index < num_of_days: # If enough data then add real data
                            row_to_add = list(line[row_to_add_index])
                            row_to_add_date_str = row_to_add[date_index]
                            row_to_add[date_index] = int(time.mktime(time.strptime(row_to_add[date_index], _date_format))) - time.timezone
                            f_future_index = j - 1 # 1 day in future
                            f_future = line[f_future_index]
                            p_yesterday = get_previous_day(line, row_to_add_index, num_of_days, _dummy_day, 1)
                            p_two_days_ago = get_previous_day(line, row_to_add_index, num_of_days, _dummy_day, 2)

                            # 11 additonal rows
                            row_to_add.append(vol_dict[row_to_add_date_str]) # total volume
                            row_to_add.append(cap_dict[row_to_add_date_str]) # total market cap
                            row_to_add.append(get_high_low_spread(row_to_add, high_index, low_index)) # high low spread
                            row_to_add.append(get_volume_difference(row_to_add, p_yesterday, volume_index)) # volume difference from yesterday
                            row_to_add.append(get_moving_avg(line, 3, row_to_add_index, close_index)) # 3 day moving average
                            row_to_add.append(get_moving_avg(line, 5, row_to_add_index, close_index)) # 5 day moving average
                            row_to_add.append(get_moving_avg(line, 7, row_to_add_index, close_index)) # 7 day moving average
                            p_change = get_price_change_from_yesterday(row_to_add, p_yesterday, close_index)
                            p_change_two = get_price_change_from_two_days_ago(row_to_add, p_two_days_ago, close_index)
                            row_to_add.append(p_change[0]) # price change from yesterday
                            row_to_add.append(p_change[1]) # percent change from yesterday
                            row_to_add.append(p_change_two[0]) # price change from 2 days ago
                            row_to_add.append(p_change_two[1]) # percent change from 2 days ago
                            row_to_add.append(get_price_movement_yesterday(row_to_add, p_yesterday, close_index)) # get price movement value from yesterday
                            row_to_add.append(get_price_movement_two_days_ago(row_to_add, p_two_days_ago, close_index)) # get price movement value from two days ago
                            row_to_add.append(get_will_increase(f_future, row_to_add, high_index, close_index)) # get future will increase


                            # Additional fields                            
                            #p_high_low_spread = get_high_low_spread(row_to_add, high_index, low_index)
                            #p_volume_diff = get_volume_difference(row_to_add, p_yesterday, volume_index)
                            #f_will_increase = get_will_increase(f_future, row_to_add, high_index, close_index)
                            #row_to_add.append(p_high_low_spread)
                            #row_to_add.append(p_volume_diff)
                            #row_to_add.append(f_will_increase)
                        else:
                            previous_dummy_date = (today_date - timedelta(days=j)) #.strftime(_date_format)
                            prev_time = previous_dummy_date.strftime(_date_format) # dummay date
                            row_to_add[date_index] = int(time.mktime(time.strptime(prev_time, _date_format))) - time.timezone
                            #row_to_add[date_index] = previous_dummy_date.strftime(_date_format)                            
                            
                            # 11 dummy fields added to dummy record
                            row_to_add.append(0) # Dummy total volume
                            row_to_add.append(0) # Dummy total market cap
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread
                            row_to_add.append(0) # Dummy p_high_low_spread

                        del row_to_add[0]
                        del row_to_add[0]
                        del row_to_add[7]
                        del row_to_add[7]

                        today = today + row_to_add

                    out_file.write("\t".join(str(x) for x in today) + "\n")
                    if not training:
                        break

            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
    

if __name__ == "__main__":
    main()
    

    

