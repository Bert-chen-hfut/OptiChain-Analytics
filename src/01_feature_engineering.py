# src/01_feature_engineering.py
import pandas as pd
import numpy as np
import os
import joblib

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))

def run_leakproof_feature_pipeline(data_path="data/raw", processed_path="data/processed"):
    print("=" * 80)
    print(">>> [Phase 1: 严格防泄漏因果时序管道] 启动...")
    print("=" * 80)

    orders = pd.read_csv(f"{data_path}/olist_orders_dataset.csv")
    items = pd.read_csv(f"{data_path}/olist_order_items_dataset.csv")
    products = pd.read_csv(f"{data_path}/olist_products_dataset.csv")
    customers = pd.read_csv(f"{data_path}/olist_customers_dataset.csv")
    sellers = pd.read_csv(f"{data_path}/olist_sellers_dataset.csv")
    geo = pd.read_csv(f"{data_path}/olist_geolocation_dataset.csv")
    payments = pd.read_csv(f"{data_path}/olist_order_payments_dataset.csv")

    orders = orders[orders['order_status'] == 'delivered'].copy()
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c])
        
    orders = orders.dropna(subset=['order_delivered_customer_date', 'order_approved_at'])
    orders = orders.sort_values('order_purchase_timestamp').reset_index(drop=True)

    # 1. 严格时序切分点 (前 80% 训练，后 20% 盲测)
    split_idx = int(len(orders) * 0.8)
    split_date = orders.loc[split_idx, 'order_purchase_timestamp']
    print(f"--> [因果时序隔离锚点]: {split_date}")

    train_orders_meta = orders.iloc[:split_idx].copy()
    test_orders_meta = orders.iloc[split_idx:].copy()

    # 2. 仅从训练集构建卖家历史出库耗时字典 (严防 Target Leakage)
    train_items_time = items.merge(
        train_orders_meta[['order_id', 'order_approved_at', 'order_delivered_carrier_date']], 
        on='order_id'
    )
    train_items_time['dispatch_lag'] = (train_items_time['order_delivered_carrier_date'] - train_items_time['order_approved_at']).dt.total_seconds() / 86400.0
    train_items_time = train_items_time[train_items_time['dispatch_lag'] >= 0]
    
    # 卖家先验统计量
    seller_dispatch_priors = train_items_time.groupby('seller_id')['dispatch_lag'].median().to_dict()
    global_median_dispatch = train_items_time['dispatch_lag'].median()
    print(f"--> [训练集学习完成]: 提取卖家画像 {len(seller_dispatch_priors):,} 家，全局默认基准 {global_median_dispatch:.2f} 天")

    # 3. 基础空间特征准备
    geo_unique = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    customers = customers.merge(geo_unique, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    sellers = sellers.merge(geo_unique, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

    # 4. 封装统一的映射函数
    def process_split(order_df, is_train=True):
        df = order_df.copy()
        df['is_delayed'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)
        df['estimated_sla_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0

        # 商品维度聚合
        items_sub = items[items['order_id'].isin(df['order_id'])].copy()
        items_sub = items_sub.merge(products, on='product_id', how='left')
        items_sub = items_sub.merge(sellers[['seller_id', 'seller_state', 'geolocation_lat', 'geolocation_lng']], on='seller_id', how='left')
        items_sub.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
        
        # 严格使用训练集的 Prior 映射
        items_sub['seller_avg_dispatch_lag_days'] = items_sub['seller_id'].map(seller_dispatch_priors).fillna(global_median_dispatch)
        items_sub['volume_cm3'] = items_sub['product_length_cm'] * items_sub['product_height_cm'] * items_sub['product_width_cm']

        agg_items = items_sub.groupby('order_id').agg({
            'order_item_id': 'count',
            'price': 'sum',
            'freight_value': 'sum',
            'product_weight_g': 'sum',
            'volume_cm3': 'sum',
            'seller_avg_dispatch_lag_days': 'max',
            'seller_lat': 'mean',
            'seller_lng': 'mean',
            'seller_state': 'first'
        }).reset_index().rename(columns={'order_item_id': 'item_count'})

        pay_sub = payments[payments['order_id'].isin(df['order_id'])].groupby('order_id').agg({
            'payment_value': 'sum',
            'payment_installments': 'max'
        }).reset_index()

        df = df.merge(agg_items, on='order_id').merge(
            customers[['customer_id', 'customer_state', 'geolocation_lat', 'geolocation_lng']].rename(
                columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}), on='customer_id'
        ).merge(pay_sub, on='order_id', how='left')

        # 因果衍生
        df['geo_distance_km'] = haversine_np(df['cust_lng'], df['cust_lat'], df['seller_lng'], df['seller_lat'])
        df['geo_distance_km'] = df['geo_distance_km'].fillna(df['geo_distance_km'].median())
        df['is_interstate'] = (df['customer_state'] != df['seller_state']).astype(int)
        df['speed_urgency_km_per_day'] = df['geo_distance_km'] / (df['estimated_sla_days'] + 1e-3)
        df['dispatch_sla_ratio'] = df['seller_avg_dispatch_lag_days'] / (df['estimated_sla_days'] + 1e-3)
        df['product_weight_g_log'] = np.log1p(df['product_weight_g'].fillna(df['product_weight_g'].median()))
        df['volume_cm3_log'] = np.log1p(df['volume_cm3'].fillna(df['volume_cm3'].median()))
        df['freight_ratio'] = df['freight_value'] / (df['payment_value'] + 1e-5)

        df['order_hour'] = df['order_purchase_timestamp'].dt.hour
        df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
        df['is_black_friday_season'] = ((df['order_purchase_timestamp'].dt.month == 11) & 
                                        (df['order_purchase_timestamp'].dt.day >= 20)).astype(int)

        feats = [
            'is_delayed', 'estimated_sla_days', 'speed_urgency_km_per_day', 'dispatch_sla_ratio',
            'geo_distance_km', 'is_interstate', 'seller_avg_dispatch_lag_days',
            'item_count', 'product_weight_g_log', 'volume_cm3_log', 'freight_ratio',
            'payment_installments', 'order_hour', 'order_dayofweek', 'is_weekend', 'is_black_friday_season',
            'customer_state', 'seller_state'
        ]
        return pd.get_dummies(df[feats].dropna(), columns=['customer_state', 'seller_state'], drop_first=True)

    train_df = process_split(train_orders_meta, is_train=True)
    test_df = process_split(test_orders_meta, is_train=False)

    # 对齐训练集与测试集列名（缺少则补 0）
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    os.makedirs(processed_path, exist_ok=True)
    train_df.to_parquet(f"{processed_path}/train.parquet", index=False)
    test_df.to_parquet(f"{processed_path}/test.parquet", index=False)
    print(f"✅ 无泄漏数据管道就绪！Train: {len(train_df):,} 行, Test: {len(test_df):,} 行, 特征数: {train_df.shape[1]-1}")

if __name__ == "__main__":
    run_leakproof_feature_pipeline()