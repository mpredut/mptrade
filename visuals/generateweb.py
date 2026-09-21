import os
import json

# Initial coin list (top 10).
# coins = [
#    {"name": "BTCUSDT", "quantity": 0.5, "watch": True},
#    {"name": "TAOUSDT", "quantity": 0.5, "watch": True},
#    {"name": "ETHUSDT", "quantity": 1.5, "watch": False},
#    {"name": "BNBUSDT", "quantity": 3.0, "watch": False},
#    {"name": "SOLUSDT", "quantity": 4.0, "watch": False},
#    {"name": "ADAUSDT", "quantity": 6.0, "watch": False},
#    {"name": "DOGEUSDT", "quantity": 7.0, "watch": False},
#    {"name": "DOTUSDT", "quantity": 9.0, "watch": False},
#    {"name": "LTCUSDT", "quantity": 10.0, "watch": False}
#    {"name": "ETHUSDT", "quantity": 1.5, "watch": False}
#]
 
# coins = [
    # {"name": "BTCUSDT", "quantity": 0.5, "watch": True},
    # {"name": "TAOUSDT", "quantity": 0.5, "watch": True},
    # {"name": "ETHUSDT", "quantity": 1.5, "watch": False}
# ]

coins_empty = [
]


# Initial coin list (top 10).
coins = [
    {"name": "BTCUSDC", "quantity": 0.5, "watch": True},
    {"name": "TAOUSDC", "quantity": 0.5, "watch": True}
]

# File that stores the latest configuration.
CONFIG_FILE = "last_watch_config.json"

def citeste_config_anterioara():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"watch_list": [], "repeat_count": 0}

def salveaza_config_actuala(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f)

def trebuie_sa_scoata_sunet():
    config_anterioara = citeste_config_anterioara()
    watch_list_anterioara = config_anterioara.get("watch_list", [])
    repeat_count = config_anterioara.get("repeat_count", 0)
    
    current_watch_list = [coin["name"] for coin in coins if coin["watch"]]
    
    if current_watch_list != watch_list_anterioara:
        # The list changed; reset the counter and play the sound.
        config_nou = {"watch_list": current_watch_list, "repeat_count": 1}
        salveaza_config_actuala(config_nou)
        return True
    elif repeat_count < 3:
        # The list is unchanged, but the sound may play up to three more times.
        config_nou = {"watch_list": current_watch_list, "repeat_count": repeat_count + 1}
        salveaza_config_actuala(config_nou)
        return True
    else:
        # The list is unchanged and the three-sound limit has been exceeded.
        return False

def generate_html(coins, refresh_interval=10, base_url="https://5499-85-122-194-86.ngrok-free.app/"):
    sunet_activ = trebuie_sa_scoata_sunet()
    
    # Minimal CSS styling.
    stil_css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f4f4f4; }
        input { width: 80px; text-align: center; }
        button { padding: 8px 12px; font-size: 14px; cursor: pointer; border: none; border-radius: 4px; }
        .btn-sell { background-color: #ff4d4d; color: white; }
        .btn-buy { background-color: #4caf50; color: white; }
    </style>
    """

    # Main HTML content.
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Coins to trade</title>
        {stil_css}
        <script>
            let audioEnabled = true;
            function enableAudio() {{ audioEnabled = true; }}
            function disableAudio() {{ audioEnabled = false; }}
        </script>
    </head>
    <body>
        <button onclick="enableAudio()">Enable sound</button>
        <button onclick="disableAudio()">Disable sound</button>
        <div class="message">
            {'There are new coins to trade!' if coins else 'No coin available to trade.'}
        </div>
    """

    if sunet_activ:
        html += """
        <script>
            if (audioEnabled) {
                let audio = new Audio('/static/bip.wav');
                audio.play().catch(err => console.error("Eroare la redarea sunetului:", err));
            }
        </script>
        """

    html += "<table><thead><tr><th>Coin</th><th>Quantity</th><th>Action</th></tr></thead><tbody>"
    for coin in coins:
        if coin["watch"]:
            html += f"""
            <tr>
                <td>{coin['name']}</td>
                <td><input type="number" value="{coin['quantity']}" id="qty-{coin['name']}"></td>
                <td>
                    <button class="btn-sell" onclick="actionSell('{coin['name']}')">Sell</button>
                    <button class="btn-buy" onclick="actionBuy('{coin['name']}')">Buy</button>
                </td>
            </tr>
            """

    html += """
        </tbody></table>
        <script>
            function actionSell(coin) {
                const quantity = document.getElementById(`qty-${coin}`).value;
                fetch('{base_url}trade/sell', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: coin, amount: parseFloat(quantity) })
                })
                .then(response => response.json())
                .then(data => alert(`Sold coin quantity: data.message`))
                .catch(err => console.error('Sell error:', err));
            }
            function actionBuy(coin) {
                const quantity = document.getElementById(`qty-${coin}`).value;
                fetch('{base_url}trade/buy', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: coin, amount: parseFloat(quantity) })
                })
                .then(response => response.json())
                .then(data => alert(`Bought coin quantity: data.message`))
                .catch(err => console.error('Buy error:', err));
            }
        </script>
    </body>
    </html>
    """
    return html

# Save output.
def save_html(html, file_name="index.html"):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"File {file_name} was generated successfully!")

# Generate and save.
html_content = generate_html(coins)
save_html(html_content, "index.html")
