import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import os

sns.set_theme(style='whitegrid')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'E-Commerce Public Dataset')

@st.cache_data
def load_data():
    customers = pd.read_csv(os.path.join(DATA_DIR, 'customers_dataset.csv'))
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(DATA_DIR, 'order_items_dataset.csv'))
    order_payments = pd.read_csv(os.path.join(DATA_DIR, 'order_payments_dataset.csv'))
    order_reviews = pd.read_csv(os.path.join(DATA_DIR, 'order_reviews_dataset.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products_dataset.csv'))
    category_translation = pd.read_csv(os.path.join(DATA_DIR, 'product_category_name_translation.csv'))

    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

    main_df = orders.merge(customers, on='customer_id', how='left')

    items_agg = order_items.groupby('order_id').agg(
        total_items=('order_item_id', 'count'),
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum')
    ).reset_index()
    main_df = main_df.merge(items_agg, on='order_id', how='left')

    payments_agg = order_payments.groupby('order_id').agg(
        payment_types=('payment_type', lambda x: list(x)),
        total_payment=('payment_value', 'sum')
    ).reset_index()
    main_df = main_df.merge(payments_agg, on='order_id', how='left')

    reviews_agg = order_reviews.groupby('order_id').agg(
        review_score_mean=('review_score', 'mean'),
        review_count=('review_id', 'count')
    ).reset_index()
    main_df = main_df.merge(reviews_agg, on='order_id', how='left')

    products_with_cat = products.merge(category_translation, on='product_category_name', how='left')
    prod_sales = order_items.merge(products_with_cat, on='product_id', how='left')
    cat_sales = prod_sales['product_category_name_english'].value_counts().head(10).reset_index()
    cat_sales.columns = ['category', 'sales']

    review_dist = order_reviews['review_score'].value_counts().sort_index().reset_index()
    review_dist.columns = ['score', 'count']

    main_df['purchase_month'] = main_df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly_sales = main_df.groupby('purchase_month').agg(
        order_count=('order_id', 'count'),
        total_revenue=('total_payment', 'sum')
    ).reset_index()

    return main_df, cat_sales, review_dist, monthly_sales, customers

main_df, cat_sales, review_dist, monthly_sales, customers = load_data()

st.set_page_config(page_title='E-Commerce Dashboard', layout='wide')
st.title('📊 E-Commerce Public Dashboard')
st.markdown('Analisis data E-commerce Brazil (Olist)')

# Sidebar
st.sidebar.header('Filter')

date_min = main_df['order_purchase_timestamp'].min()
date_max = main_df['order_purchase_timestamp'].max()
date_range = st.sidebar.date_input(
    'Rentang Waktu',
    value=[date_min, date_max],
    min_value=date_min,
    max_value=date_max
)

selected_status = st.sidebar.multiselect(
    'Status Pesanan',
    options=main_df['order_status'].unique(),
    default=['delivered']
)

state_options = customers['customer_state'].unique()
selected_states = st.sidebar.multiselect(
    'State Pelanggan',
    options=sorted(state_options),
    default=[]
)

filtered_df = main_df.copy()
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered_df = filtered_df[
        (filtered_df['order_purchase_timestamp'] >= start) &
        (filtered_df['order_purchase_timestamp'] <= end)
    ]

if selected_status:
    filtered_df = filtered_df[filtered_df['order_status'].isin(selected_status)]

if selected_states:
    filtered_df = filtered_df[filtered_df['customer_state'].isin(selected_states)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total Pesanan', f'{len(filtered_df):,}')
with col2:
    total_rev = filtered_df['total_payment'].sum()
    st.metric('Total Revenue', f'R$ {total_rev:,.0f}')
with col3:
    avg_score = filtered_df['review_score_mean'].mean()
    st.metric('Rata-rata Review', f'{avg_score:.2f}' if not pd.isna(avg_score) else 'N/A')
with col4:
    avg_items = filtered_df['total_items'].mean()
    st.metric('Rata-rata Item', f'{avg_items:.2f}' if not pd.isna(avg_items) else 'N/A')

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader('Distribusi Status Pesanan')
    status_counts = filtered_df['order_status'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8D6B94', '#E9C46A', '#F4A261']
    sns.barplot(x=status_counts.values, y=status_counts.index, palette=colors[:len(status_counts)], ax=ax)
    ax.set_xlabel('Jumlah')
    ax.set_ylabel('Status')
    st.pyplot(fig)

with col2:
    st.subheader('Distribusi Skor Review')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_review = ['#C73E1D', '#F18F01', '#F18F01', '#2E86AB', '#2E86AB']
    sns.barplot(x='score', y='count', data=review_dist, palette=colors_review, ax=ax)
    ax.set_xlabel('Skor')
    ax.set_ylabel('Jumlah')
    for i, v in enumerate(review_dist['count']):
        ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)
    st.pyplot(fig)

st.subheader('10 Kategori Produk Terlaris')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='sales', y='category', data=cat_sales, palette='viridis', ax=ax)
for i, v in enumerate(cat_sales['sales']):
    ax.text(v + 50, i, str(v), va='center', fontsize=9)
ax.set_xlabel('Jumlah Penjualan')
ax.set_ylabel('Kategori')
st.pyplot(fig)

st.subheader('Tren Penjualan Bulanan')
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_sales['purchase_month'], monthly_sales['order_count'],
        marker='o', color='#2E86AB', linewidth=2)
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Pesanan')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

st.caption('Sumber: Olist Brazilian E-commerce Dataset')
