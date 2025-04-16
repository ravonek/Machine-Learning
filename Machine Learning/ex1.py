import pandas as pd

df = pd.read_json("/Users/adilkhankaldybekov/Desktop/adilkhan/Machine Learning/postcodes.json")

property_df = pd.read_csv("/Users/adilkhankaldybekov/Desktop/adilkhan/Machine Learning/price_paid_records_small 3.csv")
property_df['postcode'] = property_df['postcode'].str.replace(' ', '').str.upper()

merged = pd.merge(property_df, df[['postcode', 'longitude']], on='postcode', how='left')


total_sales = len(merged)

east_sales = len(merged[merged['longitude'] > 0])

percentage = (east_sales / total_sales) * 100

print(f"Sales of East Greenwich: {percentage:.2f}%")

