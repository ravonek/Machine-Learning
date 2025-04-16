import pandas as pd
import matplotlib.pyplot as plt

prices_df = pd.read_csv("price_paid_records_small.csv")
postcodes_df = pd.read_csv("postcodes.csv")

prices_df['Town/City'] = prices_df['Town/City'].str.strip().str.upper()
postcodes_df['town'] = postcodes_df['town'].str.strip().str.upper()
postcodes_df = postcodes_df.drop_duplicates(subset=['town'])
postcodes_df.set_index('town', inplace=True)

merged = prices_df.join(postcodes_df[['longitude', 'latitude']], on='Town/City', how='left').dropna()

plt.figure(figsize = (8, 5))

east = merged[merged['longitude'] > 0]
west = merged[merged['longitude'] <= 0]

plt.scatter(east['longitude'], east['latitude'], c = 'g', s = 30, label = "East of Greenwich")
plt.scatter(west['longitude'], west['latitude'], c = 'r', s = 30, label = "West of Greenwich")
plt.scatter(0,51.48162, c = 'k', s = 300, marker = 'o' ,label = "Greenwich")


#Just an option for better experience
plt.title("Property of Sales East/West of Greenwich")
plt.xlabel("Latitude", fontsize = 35)
plt.ylabel("Longitude", fontsize = 35)
plt.legend()
plt.grid(True)

plt.show()


