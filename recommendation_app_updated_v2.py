
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the initial dataset
data = pd.read_csv('Sample_File!.csv')

# Function to process the data and compute similarities
def process_data(data):
    # Remove outliers and unwanted data
    data = data[data['Price'] > 0]  # Filter out products with zero price
    data = data[data['Price'] >= 0]  # Filter out negative prices
    data = data[~data['Description'].str.contains('Test', case=False, na=False)]  # Filter out 'Test' in Description
    data = data[~data['StockCode'].str.contains('TEST', na=False)]  # Filter out 'TEST' in StockCode

    user_item_matrix = data.pivot_table(index='Customer ID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

item_similarity_df = process_data(data)

# Recommendation function
def recommend_products(stockcode):
    similar_products = item_similarity_df[stockcode].sort_values(ascending=False)
    similar_products = similar_products.drop(stockcode)
    cross_sell_product = similar_products.idxmax()
    current_price = data[data['StockCode'] == stockcode]['Price'].mean()
    
    # Upselling logic: products between 1.5x to 3x of the current price
    upscale_candidates = data[(data['Price'] > 1.5 * current_price) & (data['Price'] <= 3 * current_price) & (data['StockCode'] != stockcode)]
    if not upscale_candidates.empty:
        upscale_product = upscale_candidates.groupby('StockCode')['Price'].mean().idxmax()
    else:
        upscale_product = similar_products.drop(cross_sell_product).idxmax()
    return cross_sell_product, upscale_product

# Streamlit app
st.title("Product Recommendation System")
st.write("Select a product to get recommendations:")

# Dropdown for product selection
product_display = data.drop_duplicates(subset='StockCode')[['StockCode', 'Description', 'Price']]
product_display['display'] = product_display['StockCode'] + " - " + product_display['Description'] + " (" + product_display['Price'].astype(str) + ")"
selected_product_display = st.selectbox('Choose a product', product_display['display'].tolist())
selected_product = selected_product_display.split(" - ")[0]

# Upload additional data
uploaded_file = st.file_uploader("Upload more past data", type=['csv'])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    data = pd.concat([data, new_data], ignore_index=True)
    st.write("Data uploaded successfully!")
    if st.button('Process Uploaded Data'):
        item_similarity_df = process_data(data)
        st.write("Uploaded data processed successfully!")

# Display recommendations
cross_sell, upscale = recommend_products(selected_product)
cross_sell_info = data[data['StockCode'] == cross_sell].iloc[0]
upscale_info = data[data['StockCode'] == upscale].iloc[0]
st.write(f"Recommended product for cross-selling: {cross_sell_info['StockCode']} - {cross_sell_info['Description']} (Price: {cross_sell_info['Price']})")
st.write(f"Recommended product for upselling: {upscale_info['StockCode']} - {upscale_info['Description']} (Price: {upscale_info['Price']})")

# Display confidence score
confidence_score = item_similarity_df.loc[selected_product, cross_sell] * 100
st.write(f"Confidence Score for Recommendation: {confidence_score:.2f}%")

st.write("## Explanation:")
st.write(f"The product {cross_sell} is recommended for cross-selling because it's frequently bought together with the selected product.")
st.write(f"The product {upscale} is recommended for upselling based on the modified price criteria.")
