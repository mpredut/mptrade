"""Legacy CoinMarketCap/Binance comparison script.

This module performs paid/public HTTP requests, sleeps, builds data frames, and
prints reports immediately when imported; it is not a side-effect-free library.
The request paths below do not set timeouts or call ``raise_for_status``.
"""

import math
import os
import time
import requests
import pandas as pd
from botcore import load_env_stack, safe_nonneg_int, safe_pos_float

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_env_stack(os.path.join(_ROOT, ".env"))

# CoinMarketCap credential sourced from environment (CMC_API_KEY)
API_KEY_CMC = os.environ.get("CMC_API_KEY", "").split("#", 1)[0].strip()
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
    max_pages = 50
    pages = 0

    while pages < max_pages:
        pages += 1
        params_cmc = {
            'start': str(start),
            'limit': str(limit),
            'convert': 'USD'
        }
        time.sleep(5)
        try:
            response = requests.get(url_cmc, headers=headers_cmc, params=params_cmc, timeout=15)
        except requests.RequestException:
            break
        if getattr(response, "status_code", 200) != 200:
            break
        try:
            data = response.json()
        except ValueError:
            break
        if not isinstance(data, dict) or not isinstance(data.get('data'), list):
            break

        # Convert the current response page to local rows.
        for coin in data['data']:
            if not isinstance(coin, dict):
                continue
            name = coin.get('name')
            symbol = coin.get('symbol')
            if not symbol:
                continue
            launch_date = pd.to_datetime(coin.get('date_added'), errors='coerce')
            quote = coin.get('quote')
            usd_quote = quote.get('USD') if isinstance(quote, dict) else None
            if not isinstance(usd_quote, dict):
                usd_quote = {}
            price = usd_quote.get('price')
            website_slug = coin.get('slug')  # Retained as an additional identifier.
            change_24h = usd_quote.get('percent_change_24h')
            change_7d = usd_quote.get('percent_change_7d')
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
    return pd.DataFrame(
        coins,
        columns=[
            "name",
            "symbol",
            "launch_date",
            "price",
            "website_slug",
            "change_24h",
            "change_7d",
        ],
    )

# Fetch all Binance ticker prices and strip trailing ``USDT`` suffix from symbols.
def get_binance_coins():
    url_binance = "https://api.binance.com/api/v3/ticker/price"
    try:
        response = requests.get(url_binance, timeout=15)
    except requests.RequestException:
        return {}
    if getattr(response, "status_code", 200) != 200:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {}
    if not isinstance(data, list):
        return {}
    
    # Build the symbol-to-price mapping returned to the comparison routine.
    binance_data = {}
    for coin in data:
        if not isinstance(coin, dict) or 'symbol' not in coin or 'price' not in coin:
            continue
        price = safe_pos_float(coin.get('price'), default=None)
        if price is None:
            continue
        symbol = str(coin['symbol']).removesuffix('USDT')
        binance_data[symbol] = price
    return binance_data

# Match both data sources by symbol and reject price differences of five percent or more.
def find_common_coins_and_sort(topn, df_cmc, binance_data):
    # Keep CoinMarketCap symbols that exist in the Binance-derived mapping.
    common_coins = []
    topn_i = safe_nonneg_int(topn, default=10)
    
    for index, row in df_cmc.iterrows():
        symbol = row['symbol']
        price_cmc = row['price']
        
        if isinstance(binance_data, dict) and symbol in binance_data:
            price_binance = binance_data[symbol]
            px_cmc_f = safe_pos_float(price_cmc, default=None)
            px_bin_f = safe_pos_float(price_binance, default=None)
            if px_cmc_f is None or px_bin_f is None:
                continue
            # Use price proximity as a weak cross-source identity check.
            if abs(px_cmc_f - px_bin_f) / px_cmc_f < 0.05:  # Toleranta de 5%
                common_coins.append({
                    "name": row['name'],
                    "symbol": symbol,
                    "launch_date": row['launch_date'],
                    "price_cmc": px_cmc_f,
                    "price_binance": px_bin_f,
                    "change_24h": row['change_24h'],
                    "change_7d": row['change_7d'],
                    "website_slug": row['website_slug']
                })

    df_common = pd.DataFrame(
        common_coins,
        columns=[
            "name",
            "symbol",
            "launch_date",
            "price_cmc",
            "price_binance",
            "change_24h",
            "change_7d",
            "website_slug",
        ],
    )

    # Produce the requested ranked views.
    df_sorted_new = df_common.sort_values(by="launch_date", ascending=False).head(topn_i)
    
    df_sorted_greatest_increase_7d = df_common.sort_values(by="change_7d", ascending=False).head(topn_i)
    df_sorted_greatest_decrease_7d = df_common.sort_values(by="change_7d", ascending=True).head(topn_i)
    df_sorted_greatest_increase_24h = df_common.sort_values(by="change_24h", ascending=False).head(topn_i)
    df_sorted_greatest_decrease_24h = df_common.sort_values(by="change_24h", ascending=True).head(topn_i)
    
    return df_sorted_new, df_sorted_greatest_increase_7d, df_sorted_greatest_decrease_7d, df_sorted_greatest_increase_24h, df_sorted_greatest_decrease_24h




def main():
    df_cmc = get_coinmarketcap_coins()
    binance_coins = get_binance_coins()

    # Rank the newest listings.
    nb = 100
    df_sorted_new = df_cmc.sort_values(by="launch_date", ascending=False).head(nb)

    print("Cea mai noua moneda lansata pe CoinMarketCap:")
    if not df_sorted_new.empty:
        print(df_sorted_new.iloc[0])

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


if __name__ == "__main__":
    main()
