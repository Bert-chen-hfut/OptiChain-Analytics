import pandas as pd
import numpy as np
import os

def haversine_np(lon1, lat1, lon2, lat2):
    """向量化计算两点间的大圆物理距离 (km)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def build_feature_pipeline(data_path="data/raw"):
    print("=" * 70)
    print(">>> [Phase 1 增强版] 启动因果特征工程 (注入承诺时效基准)...")
    print("=" * 70)

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

    # 1. 目标变量 Y
    orders['is_delayed'] = (orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']).astype(int)
    
    # 2. 关键业务特征: 平台在下单时刻给出的承诺总天数 (SLA Time Budget)
    orders['estimated_sla_days'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 86400.0

    # 3. 经纬度映射
    geo_unique = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    customers = customers.merge(geo_unique, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    customers.rename(columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}, inplace=True)
    sellers = sellers.merge(geo_unique, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    sellers.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)

    # 4. 卖家历史出库平均耗时
    items_time = items.merge(orders[['order_id', 'order_approved_at', 'order_delivered_carrier_date']], on='order_id')
    items_time['dispatch_lag_days'] = (items_time['order_delivered_carrier_date'] - items_time['order_approved_at']).dt.total_seconds() / 86400.0
    items_time = items_time[items_time['dispatch_lag_days'] >= 0]
    seller_profile = items_time.groupby('seller_id')['dispatch_lag_days'].median().reset_index()
    seller_profile.rename(columns={'dispatch_lag_days': 'seller_avg_dispatch_lag_days'}, inplace=True)

    # 5. 一单多物聚合
    items_full = items.merge(products, on='product_id', how='left')
    items_full = items_full.merge(sellers[['seller_id', 'seller_state', 'seller_lat', 'seller_lng']], on='seller_id', how='left')
    items_full = items_full.merge(seller_profile, on='seller_id', how='left')
    items_full['seller_avg_dispatch_lag_days'] = items_full['seller_avg_dispatch_lag_days'].fillna(items_full['seller_avg_dispatch_lag_days'].median())
    items_full['volume_cm3'] = items_full['product_length_cm'] * items_full['product_height_cm'] * items_full['product_width_cm']

    order_items_agg = items_full.groupby('order_id').agg({
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

    pay_agg = payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max'
    }).reset_index()

    # 6. 合并全量宽表
    df = orders[['order_id', 'customer_id', 'order_purchase_timestamp', 'estimated_sla_days', 'is_delayed']].merge(order_items_agg, on='order_id')
    df = df.merge(customers[['customer_id', 'customer_state', 'cust_lat', 'cust_lng']], on='customer_id')
    df = df.merge(pay_agg, on='order_id', how='left')

    # 7. 强因果衍生特征
    df['geo_distance_km'] = haversine_np(df['cust_lng'], df['cust_lat'], df['seller_lng'], df['seller_lat'])
    df['geo_distance_km'] = df['geo_distance_km'].fillna(df['geo_distance_km'].median())
    df['is_interstate'] = (df['customer_state'] != df['seller_state']).astype(int)
    
    # 核心交互特征: 运距与承诺时效的比率 (km/day，时效紧迫度)
    df['speed_urgency_km_per_day'] = df['geo_distance_km'] / (df['estimated_sla_days'] + 1e-3)
    # 核心时序特征: 卖家出库预期占总 SLA 预算的比例
    df['dispatch_sla_ratio'] = df['seller_avg_dispatch_lag_days'] / (df['estimated_sla_days'] + 1e-3)

    df['product_weight_g_log'] = np.log1p(df['product_weight_g'].fillna(df['product_weight_g'].median()))
    df['volume_cm3_log'] = np.log1p(df['volume_cm3'].fillna(df['volume_cm3'].median()))
    df['freight_ratio'] = df['freight_value'] / (df['payment_value'] + 1e-5)

    df['order_hour'] = df['order_purchase_timestamp'].dt.hour
    df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    df['is_black_friday_season'] = ((df['order_purchase_timestamp'].dt.month == 11) & 
                                    (df['order_purchase_timestamp'].dt.day >= 20)).astype(int)

    features_to_keep = [
        'is_delayed', 'order_purchase_timestamp',
        'estimated_sla_days', 'speed_urgency_km_per_day', 'dispatch_sla_ratio',
        'geo_distance_km', 'is_interstate', 'seller_avg_dispatch_lag_days',
        'item_count', 'product_weight_g_log', 'volume_cm3_log', 'freight_ratio',
        'payment_installments', 'order_hour', 'order_dayofweek', 'is_weekend', 'is_black_friday_season',
        'customer_state', 'seller_state'
    ]
    df = df[features_to_keep].dropna()
    df = pd.get_dummies(df, columns=['customer_state', 'seller_state'], drop_first=True)

    # 8. 严格因果时序切分
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].drop(columns=['order_purchase_timestamp'])
    test_df = df.iloc[split_idx:].drop(columns=['order_purchase_timestamp'])

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/train.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)
    print("=" * 70)
    print(f"🎉 增强特征工程处理完毕！Train: {len(train_df):,} 行, Test: {len(test_df):,} 行, 特征数: {train_df.shape[1]-1}")
    print("=" * 70)

if __name__ == "__main__":
    build_feature_pipeline()
