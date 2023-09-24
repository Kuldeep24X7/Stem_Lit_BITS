
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets from the three smaller CSV files and combine them using relative paths
data_parts = [pd.read_csv(f"Main_Excel_File_Part_{i}.csv") for i in range(1, 4)]
data = pd.concat(data_parts, ignore_index=True)

# Data preprocessing steps and rest of the application code remain unchanged
# ... [rest of the code remains the same]



# Data preprocessing steps
data = data[data['Quantity'] > 0]  # Remove negative quantities
data = data[~data['StockCode'].str.contains('TEST|Manual', case=False, na=False)]  # Remove 'TEST' or 'Manual' in StockCode
data = data[~data['Description'].str.contains('TEST|Manual', case=False, na=False)]  # Remove 'TEST' or 'Manual' in Description
data = data[data['Price'] > 0]  # Remove rows where price is 0

# Function to process the data and compute similarities
def process_data(data):
    user_item_matrix = data.pivot_table(index='Customer ID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

item_similarity_df = process_data(data)

# Discount logic
def apply_discount(price):
    if price > 5:
        return price * 0.90
    else:
        return price * 0.975

# Recommendation function
def recommend_products(stockcode):
    similar_products = item_similarity_df[stockcode].sort_values(ascending=False)
    similar_products = similar_products.drop(stockcode)
    cross_sell_product = similar_products.idxmax()
    current_price = data[data['StockCode'] == stockcode]['Price'].mean()
    upscale_candidates = data[(data['Price'] > current_price) & (data['StockCode'] != stockcode)]
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
product_display['display'] = product_display['StockCode'] + " - " + product_display['Description'] + " (€" + product_display['Price'].astype(str) + ")"
selected_product_display = st.selectbox('Choose a product', product_display['display'].tolist())
selected_product = selected_product_display.split(" - ")[0]

# Display recommendations
cross_sell, upscale = recommend_products(selected_product)
cross_sell_info = data[data['StockCode'] == cross_sell].iloc[0]
upscale_info = data[data['StockCode'] == upscale].iloc[0]

cross_sell_discounted = apply_discount(cross_sell_info['Price'])
upscale_discounted = apply_discount(upscale_info['Price'])

st.write(f"Recommended product for cross-selling: {cross_sell_info['StockCode']} - {cross_sell_info['Description']} (Original Price: €{cross_sell_info['Price']}, Discounted Price: €{cross_sell_discounted:.2f})")
st.write(f"Recommended product for upselling: {upscale_info['StockCode']} - {upscale_info['Description']} (Original Price: €{upscale_info['Price']}, Discounted Price: €{upscale_discounted:.2f})")

# Display confidence score
confidence_score = item_similarity_df.loc[selected_product, cross_sell] * 100
st.write(f"Confidence Score for Recommendation: {confidence_score:.2f}%")

st.write("## Explanation:")
st.write(f"The cross-selling recommendation, product {cross_sell}, is based on its high similarity with the selected product. Customers who bought the selected product often also bought this recommended product.")
st.write(f"The upselling recommendation, product {upscale}, suggests a potential upgrade based on its higher price compared to the selected product. This recommendation aims to introduce customers to premium alternatives.")

