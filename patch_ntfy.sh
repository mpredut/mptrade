cat << 'INNER' >> ~/mptrade/.env

# --- Missing NTFY topics restored from backup ---
NTFY_TOPIC_TRADES=ntfy-trades-b50189
NTFY_TOPIC_GUARD=ntfy-guard-8a35d7
NTFY_TOPIC_PRICE=ntfy-price-85a945
NTFY_TOPIC_ERROR=ntfy-error-941582
INNER
sudo systemctl restart binance.service
