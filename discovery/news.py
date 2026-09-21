"""Legacy CoinMarketCap/Binance comparison script.

This module performs paid/public HTTP requests, sleeps, builds data frames, and
prints reports immediately when imported; it is not a side-effect-free library.
The request paths below do not set timeouts or call ``raise_for_status``.
"""

import time
import requests
import pandas as pd

# Embedded CoinMarketCap credential used directly by this script.
API_KEY_CMC = "4d587781-722b-40a3-83f0-2436d45942f7"
url_cmc = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"


# Headers sent with every CoinMarketCap request.
headers_cmc = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY_CMC,
}

# Fetch paginated CoinMarketCap listings until the response lacks data.
def get_coinmarketcap_coins():
    coins = []
    start = 1  # Start with the first listing.
    limit = 1000  # Requested page size.

    while True:
        params_cmc = {
            'start': str(start),
            'limit': str(limit),
            'convert': 'USD'
        }
        time.sleep(5)
        response = requests.get(url_cmc, headers=headers_cmc, params=params_cmc)
        data = response.json()
        if data is None or 'data' not in data:
            break

        # Convert the current response page to local rows.
        for coin in data['data']:
            name = coin['name']
            symbol = coin['symbol']
            launch_date = pd.to_datetime(coin['date_added'])
            price = coin['quote']['USD']['price']
            website_slug = coin['slug']  # Retained as an additional identifier.
            change_24h = coin['quote']['USD']['percent_change_24h']
            change_7d = coin['quote']['USD']['percent_change_7d']
            coins.append({
                "name": name,
                "symbol": symbol,
                "launch_date": launch_date,
                "price": price,
                "website_slug": website_slug,
                "change_24h": change_24h,
                "change_7d": change_7d
            })

        if not data['data']:  # Stop after an empty page.
            break
        start += limit  # Advance to the next page.

    print(f"Am extras {len(coins)} monezi")
    return pd.DataFrame(coins)

# Fetch all Binance ticker prices and strip every ``USDT`` substring from symbols.
def get_binance_coins():
    url_binance = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url_binance)
    data = response.json()
    
    # Build the symbol-to-price mapping returned to the comparison routine.
    binance_data = {}
    for coin in data:
        symbol = coin['symbol'].replace('USDT', '')  # Current code is not suffix-specific.
        price = float(coin['price'])
        binance_data[symbol] = price
    return binance_data

# Match both data sources by symbol and reject price differences of five percent or more.
def find_common_coins_and_sort(topn, df_cmc, binance_data):
    # Keep CoinMarketCap symbols that exist in the Binance-derived mapping.
    common_coins = []
    
    for index, row in df_cmc.iterrows():
        symbol = row['symbol']
        price_cmc = row['price']
        
        if symbol in binance_data:
            price_binance = binance_data[symbol]
            # Use price proximity as a weak cross-source identity check.
            if abs(price_cmc - price_binance) / price_cmc < 0.05:  # Toleranta de 5%
                common_coins.append({
                    "name": row['name'],
                    "symbol": symbol,
                    "launch_date": row['launch_date'],
                    "price_cmc": price_cmc,
                    "price_binance": price_binance,
                    "change_24h": row['change_24h'],
                    "change_7d": row['change_7d'],
                    "website_slug": row['website_slug']
                })

    df_common = pd.DataFrame(common_coins)

    # Produce the requested ranked views.
    df_sorted_new = df_common.sort_values(by="launch_date", ascending=False).head(topn)
    
    df_sorted_greatest_increase_7d = df_common.sort_values(by="change_7d", ascending=False).head(topn)
    df_sorted_greatest_decrease_7d = df_common.sort_values(by="change_7d", ascending=True).head(topn)
    df_sorted_greatest_increase_24h = df_common.sort_values(by="change_24h", ascending=False).head(topn)
    df_sorted_greatest_decrease_24h = df_common.sort_values(by="change_24h", ascending=True).head(topn)
    
    return df_sorted_new, df_sorted_greatest_increase_7d, df_sorted_greatest_decrease_7d, df_sorted_greatest_increase_24h, df_sorted_greatest_decrease_24h




# The executable report begins here and also runs on import.
df_cmc = get_coinmarketcap_coins()
binance_coins = get_binance_coins()
    

# Rank the newest listings.
nb = 100
df_sorted_new = df_cmc.sort_values(by="launch_date", ascending=False).head(nb)

print("Cea mai noua moneda lansata pe CoinMarketCap:")
print(df_sorted_new.iloc[0])  # Raises if the fetched data frame is empty.

# Rank gains and losses over the two requested periods.
df_sorted_greatest_increase_7d = df_cmc.sort_values(by="change_7d", ascending=False).head(10)
df_sorted_greatest_decrease_7d = df_cmc.sort_values(by="change_7d", ascending=True).head(10)
df_sorted_greatest_increase_24h = df_cmc.sort_values(by="change_24h", ascending=False).head(10)
df_sorted_greatest_decrease_24h = df_cmc.sort_values(by="change_24h", ascending=True).head(10)

# Print all report sections.
print(f"Primele {nb} monede noi:")
pd.set_option('display.max_rows', 100)  # Display up to 100 rows.
print(df_sorted_new)

print("\nTop 10 cresteri pe 7 zile:")
print(df_sorted_greatest_increase_7d)

print("\nTop 10 scaderi pe 7 zile:")
print(df_sorted_greatest_decrease_7d)

print("\nTop 10 cresteri pe 24 de ore:")
print(df_sorted_greatest_increase_24h)

print("\nTop 10 scaderi pe 24 de ore:")
print(df_sorted_greatest_decrease_24h)

###########
# Build and print the cross-source comparison.
df_top_10_new, df_top_10_increase_7d, df_top_10_decrease_7d, df_top_10_increase_24h, df_top_10_decrease_24h = find_common_coins_and_sort(10, df_cmc, binance_coins)

print("The first 10 coins available on both CoinMarketCap and Binance, sorted by newness:")
print(df_top_10_new)

print("\nTop 10 cresteri pe 7 zile:")
print(df_top_10_increase_7d)

print("\nTop 10 scaderi pe 7 zile:")
print(df_top_10_decrease_7d)

print("\nTop 10 cresteri pe 24 de ore:")
print(df_top_10_increase_24h)

print("\nTop 10 scaderi pe 24 de ore:")
print(df_top_10_decrease_24h)
