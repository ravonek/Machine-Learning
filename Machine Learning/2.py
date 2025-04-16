import pandas as pd
import matplotlib.pyplot as plt

prices_df = pd.read_csv("price_paid_records_small.csv")
postcodes_df = pd.read_csv("postcodes.csv")

prices_df['Town/City'] = prices_df['Town/City'].str.strip().str.upper()
postcodes_df['town'] = postcodes_df['town'].str.strip().str.upper()
postcodes_df = postcodes_df.drop_duplicates(subset=['town'])
postcodes_df.set_index('town', inplace=True)

merged = prices_df.join(postcodes_df[['longitude', 'latitude']], on='Town/City', how='left').dropna()

plt.figure(figsize=(12, 8))

east = merged[merged['longitude'] > 0]
west = merged[merged['longitude'] <= 0]

plt.scatter(east['longitude'], east['latitude'], c='green', s=10, label='East of Greenwich')
plt.scatter(west['longitude'], west['latitude'], c='red', s=10, label='West of Greenwich')
plt.scatter(0, 51.4769, c='black', s=100, marker='o', label='Greenwich')

plt.axvline(x=0, color='gray', linestyle='--')
plt.title('Property Sales East/West of Greenwich')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)

plt.xlim(-6, 2)
plt.ylim(50, 57)

plt.show()