
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Initial dataset loading
data = pd.read_csv('Sample_File!.csv')

# Preprocessing function
def preprocess_data(data):
    # Remove negative quantities
    data = data[data['Quantity'] > 0]
    # Remove zero prices
    data = data[data['Price'] > 0]
    return data

# Apriori processing function
def process_data_for_apriori(data):
    basket = data.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket_sets = (basket > 0).astype(int)
    frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules

# Recommendation function using Apriori
def recommend_products_apriori(product_description, rules):
    product_rules = rules[rules['antecedents'] == frozenset({product_description})]
    product_rules = product_rules.sort_values(by='confidence', ascending=False)
    recommendations = product_rules['consequents'].values
    confidences = product_rules['confidence'].values
    results = []
    for rec, conf in zip(recommendations, confidences):
        results.append((list(rec)[0], conf))
    return results[:2]

# Streamlit app
st.title("Product Recommendation System using Apriori")
st.write("Select a product to get recommendations:")

# Upload additional data
uploaded_file = st.file_uploader("Upload more past data (up to 200MB)", type=['csv'], accept_multiple_files=False)
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    new_data = preprocess_data(new_data)
    data = pd.concat([data, new_data], ignore_index=True)
    st.write("Data uploaded successfully. Click the 'Process Data' button to update recommendations.")
    if st.button('Process Data'):
        rules = process_data_for_apriori(data)
        st.write("Data processed successfully!")

# Dropdown for product selection
product_display = data.drop_duplicates(subset='Description')['Description']
selected_product_description = st.selectbox('Choose a product', product_display.tolist())

# Display recommendations
rules = process_data_for_apriori(data)
recommendations = recommend_products_apriori(selected_product_description, rules)
if recommendations:
    for idx, (rec, conf) in enumerate(recommendations, 1):
        st.write(f"{idx}. Recommended product: {rec} (Confidence: {conf:.2f}%)")
else:
    st.write("No strong recommendations for the selected product.")

st.write("## Explanation:")
st.write("Recommendations are based on the Apriori algorithm, which finds products that are frequently bought together. The confidence score indicates the likelihood of the recommended product being purchased after the selected product.")

