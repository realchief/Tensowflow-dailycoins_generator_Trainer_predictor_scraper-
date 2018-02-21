import tensorflow as tf
import pandas as pd
import tempfile
import numpy as np
import shutil
import os
import glob
import codecs
import csv

print("***** COIN TRAINER ******\n")

training_data = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_training_data.csv"
evaluation_data = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\daily_coins\coinmarketcap_evaluation.csv"
_feature_names = []
_default_vals = []
_label_index = 13

CSV_COLUMNS = []
USABLE_COLS = []
FEATURE_COLS = []


print("Setting default values and usable features...")
with codecs.open(training_data, 'r', "utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='~')
    for line in reader:
        CSV_COLUMNS = line
        #print("CSV_COLUMNS Count: " + str(len(CSV_COLUMNS)))
        break
    
    for col in CSV_COLUMNS:
        if col != "will_increase":
            FEATURE_COLS.append(col)
        if col == "name" or col == "symbol" or col == "tags" or col == "markets":
            _default_vals.append([str(0)])
        else:
            _default_vals.append([float(0)])


print("reading csv file into data frame...\n\n")
df = pd.read_csv(training_data, delimiter="\t", encoding = "utf-8", skiprows=1, names=CSV_COLUMNS, skipinitialspace=True)
df = df.reindex(np.random.permutation(df.index))
#print(str(df.isnull().values.any()))
#df.replace(np.NaN, "NANINAKINANA")
#df = df.fillna("NANINAKINANA")
#df = df.fillna(0)

#unique_total_vol = df['tags'].unique().tolist()
#unique_total_cap = df['total_market_cap'].unique().tolist()
#print("unique_total_vol: " + str(len(unique_total_vol)))
#print("unique_total_cap: " + str(len(unique_total_cap)))

#for val in unique_total_vol:
#    print(val)

print("\n\n")
print(df.dtypes)
print("\n\n")

print("Data Count: " + str(len(df)))
train_amount = int(0.90 * len(df))
test_amount = int(0.91 * len(df))
df_train = df[:train_amount]
df_test = df[train_amount:test_amount]

print("Training Data: " + str(len(df_train)))
print("Evaluation Data: " + str(len(df_test)))
print("\n\n")


def decode_csv(line):
    parsed_line = tf.decode_csv(line, _default_vals, field_delim="\t")    
    label = parsed_line[_label_index] # set the index for the label/output field
    del parsed_line[_label_index] # Delete label from main train data            
    features = parsed_line # Everything (but label) are the features
    #print("Last Features Count: " + str(len(FEATURE_COLS)))
    #print("Last Parsed Count: " + str(len(features)))
    d = dict(zip(FEATURE_COLS, features)), label
    return d

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):

   dataset = (tf.contrib.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels


# Features Creation
#market_name = tf.feature_column.categorical_column_with_vocabulary_list("MarketName", unique_market_names)
name = tf.feature_column.categorical_column_with_hash_bucket("name", 3000)
symbol = tf.feature_column.categorical_column_with_hash_bucket("symbol", 3000)
tags = tf.feature_column.categorical_column_with_hash_bucket("tags", 20)
markets = tf.feature_column.categorical_column_with_hash_bucket("markets", 10000)

numeric_features = []
cnt = len(CSV_COLUMNS)
for i in range(1, cnt):
    if "will_increase" != CSV_COLUMNS[i] and "name" != CSV_COLUMNS[i] and "symbol" != CSV_COLUMNS[i] and "tags" != CSV_COLUMNS[i] and "markets" != CSV_COLUMNS[i]:
    #if i != 9 and i != 10:
        numeric_features.append(tf.feature_column.numeric_column(CSV_COLUMNS[i]))

volume_buckets = tf.feature_column.bucketized_column(numeric_features[5], boundaries=[10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000])
market_cap_buckets = tf.feature_column.bucketized_column(numeric_features[6], boundaries=[1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000])

crossed_columns = [
    tf.feature_column.crossed_column([volume_buckets, market_cap_buckets], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["name", market_cap_buckets], hash_bucket_size=10000),
    tf.feature_column.crossed_column([volume_buckets, "name"], hash_bucket_size=10000),
    tf.feature_column.crossed_column([volume_buckets, "markets"], hash_bucket_size=1000),
    tf.feature_column.crossed_column([market_cap_buckets, "markets"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(["tags", "markets"], hash_bucket_size=1000)
]

deep_columns = [
    tf.feature_column.indicator_column(name),
    tf.feature_column.indicator_column(symbol),
    tf.feature_column.indicator_column(tags),
    tf.feature_column.embedding_column(markets, dimension=10)
]

#Add dynamic deep coloumns
for feat in numeric_features:
    deep_columns.append(feat)


#feature_columns = crossed_columns + deep_columns
#config  = feature_columns[5]._parse_example_spec
model_dir = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\temp_models"
print("\nClearing model temp directory...\n")
files = glob.glob(model_dir + "\*")
for f in files:
    os.remove(f)

print("\nBuilding model...\n")
# Set Model
search_model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=deep_columns,
    #linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.00001, l1_regularization_strength=0.001, l2_regularization_strength=0.001),
    dnn_hidden_units=[1000, 1000, 1000])


#search_model = tf.estimator.DNNLinearCombinedRegressor(
#    model_dir=model_dir,
    # wide settings
#    linear_feature_columns=crossed_columns,
#    linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.001, l1_regularization_strength=0.001, l2_regularization_strength=0.001),
    
    # deep settings
#    dnn_feature_columns=deep_columns,
#    dnn_hidden_units=[1000, 500, 100],
#    dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l1_regularization_strength=0.001, l2_regularization_strength=0.001))

#Build estimator
training_input = tf.estimator.inputs.pandas_input_fn(x=df_train, y=df_train["will_increase"], batch_size=128, num_epochs=None, shuffle=True, num_threads=10)
testing_input = tf.estimator.inputs.pandas_input_fn(x=df_test, y=df_test["will_increase"], batch_size=128, num_epochs=1, shuffle=False, num_threads=1)

print("\nTraining...\n")
#Train model
search_model.train(input_fn=training_input, steps=3000)
#search_model.train(input_fn=lambda: my_input_fn(training_data, True, 8))

print("\nEvaluating...\n")
# set steps to None to run evaluation until all data consumed.
results = search_model.evaluate(input_fn=testing_input, steps=None)
#results = search_model.evaluate(input_fn=lambda: my_input_fn(evaluation_data, True, 1))

#print("model directory = %s" % model_dir)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

#Save Model
print("\nSaving model...\n")
feature_columns = crossed_columns + deep_columns
#feature_columns[0]._parse_example_spec
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
servable_model_dir = r"C:\Users\mattest\Desktop\Projects\Machine_Learning\data\saved_models"
servable_model_path = search_model.export_savedmodel(servable_model_dir, export_input_fn)

