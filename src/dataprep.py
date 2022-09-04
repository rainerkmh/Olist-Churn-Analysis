#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import geopy.distance
import numpy as np
from os.path import join, dirname
import yaml
from datetime import timedelta
from math import isnan

def load_config(path):
    with open(join(dirname(__file__),path)) as file:
        config = yaml.safe_load(file)
    return config

def open_file(path):
    file = open(join(dirname(__file__),path), encoding="utf8")
    return file

def file_path(path):
    file_path = join(dirname(__file__),path)
    return file_path

def load_data():
    data = {
    	'df_customers':pd.read_csv(open_file(config['olist_customers_dataset'])),
    	'df_customers_geolocation':pd.read_csv(open_file(config['olist_geolocation_dataset'])),
    	'df_order_items':pd.read_csv(open_file(config['olist_order_items_dataset'])),
    	'df_payments':pd.read_csv(open_file(config['olist_order_payments_dataset'])),
    	'df_reviews':pd.read_csv(open_file(config['olist_order_reviews_dataset'])),
    	'df_orders':pd.read_csv(open_file(config['olist_orders_dataset'])),
    	'df_products':pd.read_csv(open_file(config['olist_products_dataset'])),
    	'df_sellers':pd.read_csv(open_file(config['olist_sellers_dataset'])),
    	'df_product_category_name_translation':pd.read_csv(open_file(config['product_category_name_translation']))
    	}
    
    for key in data.keys():
        if data[key].empty:
            raise Exception(f"{key} is empty! Try loading the dataset again")

    print('Data loaded!')
    return data

def prep_data(data):
    # Keep only product_id and product_category in products table
    data['df_products'] = data['df_products'][['product_id', 'product_category_name']]
    # Drop shipping_limit_date in order_items table
    data['df_order_items'] = data['df_order_items'].drop(columns=['shipping_limit_date'])
    # Drop order_approved_at, order_delivered_carrier_date in order_items table
    data['df_orders'] = data['df_orders'].drop(columns=['order_approved_at',
                                                        'order_delivered_carrier_date'])
    # Drop review_comment_title, review_creation_date, review_answer_timestamp, in reviews table
    data['df_reviews'] = data['df_reviews'].drop(columns=['review_comment_title',
                                                          'review_creation_date',
                                                          'review_answer_timestamp'])\
    # Drop geolocation_city, geolocation_state in customers_geolocation table
    data['df_customers_geolocation'] = data['df_customers_geolocation'].drop(columns=['geolocation_city', 
                                                                                      'geolocation_state'])
    
    # Translate product names from PT to EN
    product_cat_names_eng = dict(zip(data['df_product_category_name_translation'].product_category_name, 
                                     data['df_product_category_name_translation'].product_category_name_english))
    data['df_products'].replace({"product_category_name":product_cat_names_eng}, inplace = True)
    
    # Merge the order items table with the products table on product_id
    items = pd.merge(left=data['df_order_items'], right=data['df_products'], on='product_id', how='left')
    # Merge items with the sellers table on seller_id
    items = pd.merge(left=items, right=data['df_sellers'], on='seller_id', how='left')
    
    # Relabel minority categories to others
    relabel_categories = items['product_category_name'].value_counts().index[20:]
    items.loc[items['product_category_name'].isin(relabel_categories), 'product_category_name'] = 'others'

    # It is possible for one order to have multiple item types and multiple sellers.
    # Count number of items per order id
    num_items_order_id = items.groupby('order_id')['order_item_id'].transform("count")
    items = pd.merge(left=items, right=num_items_order_id, left_index=True, right_index=True)
    items = items.rename(columns={'order_item_id_x': 'order_item_id', 
                                                 'order_item_id_y': 'item_quantity'})
    
    # Find item in order with highest price
    highest_price = items.groupby('order_id', as_index = False).agg({'price':max})
    # Find total sum of price and freight respectively per order id
    sum_price_freight = items.groupby('order_id', as_index = False).agg({'price':sum, 'freight_value':sum})

    # Rename column
    highest_price = highest_price.rename(columns={'price': 'highest_price_item'})

    # Join items with highest_price table 
    items = pd.merge(left = items, right = highest_price, on = 'order_id', how = 'left')
    
    # Retain only order items per order id that contain the highest price
    items = items[items['price'] == items['highest_price_item']]
    # Remove duplicate order items if they contain the same highest price and map remaining info 
    # belonging to the first line item per order id 
    items = items.groupby('order_id', as_index = False).agg({'product_id': 'first',
                                                                               'seller_id': 'first',
                                                                               'product_category_name':'first', 
                                                                               'seller_zip_code_prefix':'first',
                                                                               'seller_city':'first',
                                                                               'seller_state':'first',
                                                                               'highest_price_item':'first',
                                                                               'item_quantity':'max'
                                                                              })
    
    # Drop highest_price_item column
    items = items.drop(columns=['highest_price_item'])
    
    # Relabel price column
    items['total_price'] = sum_price_freight['price']
    # Relabel freight_value column
    items['total_freight_value'] = sum_price_freight['freight_value']
    
    # Change columns to datetime objects
    data['df_orders']['order_estimated_delivery_date'] = pd.to_datetime(data['df_orders']['order_estimated_delivery_date'])
    data['df_orders']['order_purchase_timestamp'] = pd.to_datetime(data['df_orders']['order_purchase_timestamp'])
    data['df_orders']['order_delivered_customer_date'] = pd.to_datetime(data['df_orders']['order_delivered_customer_date'])
    
    # Group the reviews by order_d and take the average of the review scores
    # For review_id, we just take the 1st
    data['df_reviews'] =  data['df_reviews'].groupby('order_id', as_index = False)                                             .agg({'review_id': 'first',
                                                  'review_score':'mean',
                                                  'review_comment_message': lambda x: ' '.join(map(str, x))
                                                 })
    
    # Merge the orders table with the reviews table on order_id
    orders = pd.merge(left=data['df_orders'], right=data['df_reviews'], on='order_id', how='left')
    
    # To capture the total payment amount, we must sum the payment_value column
    # For the remaining columns, we simply need to keep the max values
    data['df_payments'] = data['df_payments'].groupby('order_id', as_index = False)                                              .agg({'payment_sequential': max,
                                                   'payment_installments': max,
                                                   'payment_value': sum,
                                                   'payment_type':'first'
                                                  })
    # Merge orders with the payments table on order_id
    orders = pd.merge(left=orders, right=data['df_payments'], on='order_id', how='left')
    
    # Merge orders with items on order_id
    orders = pd.merge(left=orders, right=items, on='order_id', how='left')
    
    # Rename column for table merging 
    data['df_customers_geolocation'] = data['df_customers_geolocation'].rename(columns={'geolocation_zip_code_prefix':
                                                                                        'customer_zip_code_prefix'})

    # Each zipcode may contain multiple latitude and longitude values. 
    # Hence, we find the average values to associate with a single zip code.
    data['df_customers_geolocation'] = data['df_customers_geolocation'].groupby('customer_zip_code_prefix')                                                                        .agg({'geolocation_lat': 'median',
                                                                             'geolocation_lng': 'median'})
    
    # Merge the customers table with the geolocation table on customer_zip_code_prefix
    customers = pd.merge(left=data['df_customers'], right=data['df_customers_geolocation'], 
                         on='customer_zip_code_prefix', how='left')
    
    # Rename geolocation columns
    customers = customers.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})
    
    # Merge the customers with orders on customer_id
    customers = pd.merge(left=customers, right=orders, on='customer_id', how='left')

    # Create intermediate features
    # Create feature num_orders_per_customer
    num_orders_per_customer = customers.groupby('customer_unique_id')['order_id'].nunique()
    num_orders_per_customer = num_orders_per_customer.to_dict()
    customers['num_orders_per_customer'] = customers['customer_unique_id'].map(num_orders_per_customer)
    
    # Create table for repeat customers (>1 order)
    repeat_customers = customers[customers['num_orders_per_customer'] > 1]
    repeat_customers = repeat_customers.groupby(['customer_unique_id', 'order_id'], as_index = False)                                        .agg({'order_purchase_timestamp':'first'})                                        .sort_values(['customer_unique_id','order_purchase_timestamp'], 
                                        ascending=[True,True])
    
    # Create timestamps for first and second orders for repeat customers
    first_order_timestamp = repeat_customers.groupby('customer_unique_id').nth([0])['order_purchase_timestamp']
    second_order_timestamp = repeat_customers.groupby('customer_unique_id').nth([1])['order_purchase_timestamp']
    
    # Find time difference between first and second orders
    time_diff_1st_2nd_order = (second_order_timestamp - first_order_timestamp).apply(lambda x: x.days)
    time_diff_1st_2nd_order = pd.DataFrame({'customer_unique_id':time_diff_1st_2nd_order.index, 
                                            'time_diff_1st_2nd_order':time_diff_1st_2nd_order.values})
    
    # Merge timestamps with customers table
    customers = pd.merge(left = customers, right = time_diff_1st_2nd_order, 
                         on = 'customer_unique_id', how = 'left')
    
    # Create intermediate feature 'first_purchase_timestamp'
    first_purchase = customers.groupby('customer_unique_id', as_index = False)                               .agg({'order_purchase_timestamp':min})
    first_purchase = first_purchase.rename(columns={'order_purchase_timestamp':
                                                    'first_purchase_timestamp'})
    
    # Merge customers with first purchase table
    customers = pd.merge(left=customers, right=first_purchase, on='customer_unique_id', how = 'left')
    
    # Remove customers whose first purchase falls within latest 12 months (365 days)
    churn_day_limit = config['churn_day_limit']
    last_timestamp = customers['first_purchase_timestamp'].sort_values().iat[-1]
    customers = customers[customers['first_purchase_timestamp'] < (last_timestamp - timedelta(days=churn_day_limit))]
    
    # Define time window for customer churn (i.e. no additional purchases made within x days of first purchase)
    

    # function to determine whether a customer churns or not
    def determine_repeat_customer(time_diff):
          if isnan(time_diff):
            return 0
          elif time_diff >= churn_day_limit:
            return 0
          else:
            return 1
    
    # Create 'repeat_customer' target variable. 1 means repeat customer and 0 means repeat_customer
    # Note: If time_diff_1st_2nd_order is NaN, that means no 2nd order was ever placed thus it must not be a repeat customer
    customers['repeat_customer'] = customers['time_diff_1st_2nd_order'].apply(determine_repeat_customer)
    
    # Retain only first orders made by each customer and remove the rest
    customers = customers[customers['first_purchase_timestamp'] == customers['order_purchase_timestamp']]
    
    # Rename customers to df out of convenience since we have arrived at our prepped dataset
    df = customers
    
    # Breakdown order_purchase_timestamp into year, month, day, day of week, is weekend
    df['year'] = pd.DatetimeIndex(df['order_purchase_timestamp']).year
    df['month'] = pd.DatetimeIndex(df['order_purchase_timestamp']).month
    df['day'] = pd.DatetimeIndex(df['order_purchase_timestamp']).day
    df['day_of_week'] = pd.DatetimeIndex(df['order_purchase_timestamp']).dayofweek
    df['is_weekend'] = (df['day_of_week'] > 4).astype(int)
    
    # Create new features 'purchase_to_est_delivery' and 'est_delivery_to_actual'
    # 'days_from_purchase_to_est_delivery' is the number of days 
    # between the date of purchase and the estimated delivery date
    df['purchase_to_est_delivery'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp'])                                                                            .astype('timedelta64[D]')
    # 'days_from_est_delivery_to_actual' is the number of days 
    # between the estimated delivery date and the actual delivery date
    df['est_delivery_to_actual'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date'])                                                                          .astype('timedelta64[D]')
    
    # 'days_from_purchase_to_actual' is the number of days 
    # between the date of purchase date and the actual delivery date
    df['purchase_to_actual'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp'])                                                                                 .astype('timedelta64[D]')
    
    # Reuse customer geolocation table to find seller geolocation
    sellers_geolocation = data['df_customers_geolocation'].reset_index()
    sellers_geolocation = sellers_geolocation.rename(columns={'customer_zip_code_prefix': 'seller_zip_code_prefix'})
    
    # Merge df with seller geolocation table
    df = pd.merge(left = df, right = sellers_geolocation, on = "seller_zip_code_prefix", how = "left")
    # Rename geolocation columns
    df = df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})
    
    # Function to calculate distance between geographical coordinates
    def distance(customer_lat, customer_lng, seller_lat, seller_lng):
        if any([pd.isnull(val) for val in locals().values()]):
            return None
        else:
            return geopy.distance.geodesic((customer_lat, customer_lng), (seller_lat,seller_lng)).km

    coordinate_cols = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']
    
    # Create new feature 'customer_seller_dist'
    df['customer_seller_dist'] = df[coordinate_cols].apply(lambda x: distance(x['customer_lat'],
                                                                              x['customer_lng'], 
                                                                              x['seller_lat'], 
                                                                              x['seller_lng']),
                                                                              axis=1)

    # Remove unneeded and intermediate/transitory columns
    customers = customers.drop(columns=['customer_id', 
                                        'customer_unique_id', 
                                        'order_id', 
                                        'review_id', 
                                        'product_id', 
                                        'seller_id', 
                                        'customer_zip_code_prefix',
                                        'seller_zip_code_prefix',
                                        'time_diff_1st_2nd_order', 
                                        'num_orders_per_customer', 
                                        'first_purchase_timestamp',
                                        'review_comment_message',
                                        'order_purchase_timestamp',
                                        'order_delivered_customer_date',
                                        'order_estimated_delivery_date'
                                       ])
    
    # Apply finishing touches and preview final table
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    
    # Reorder columns
    df = df[['customer_city', 'customer_state', 'seller_city', 'seller_state', 'customer_lat', 'customer_lng','seller_lat',
             'seller_lng', 'customer_seller_dist', 'review_score', 'purchase_to_est_delivery', 'est_delivery_to_actual', 
             'purchase_to_actual', 'payment_sequential', 'payment_installments', 'payment_value','payment_type',
             'total_price', 'total_freight_value', 'product_category_name', 'item_quantity', 'year', 'month', 'day',
             'day_of_week', 'is_weekend', 'order_status','repeat_customer']]

    # Standardize all null values to np.nan
    df.fillna(np.nan, inplace = True)

    # Remove duplicate data
    df = df.drop_duplicates()
    
    print('Dataframe prepared!')
    return df

config = load_config("config.yaml")

if __name__ == "__main__":
    data = load_data()
    df = prep_data(data)
    print(df.info())
    if config['save_df'] == 1:
        df.to_csv(file_path('../data/dataframe.csv'))
        print('dataframe.csv was saved into the ''data'' folder')




