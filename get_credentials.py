"""
Get your Polymarket CLOB API credentials.

Usage:
    1. Set your PRIVATE KEY in the .env file (NOT your wallet address!)
    2. Run:  python get_credentials.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")

if not private_key or private_key == "0xYOUR_PRIVATE_KEY_HERE":
    print("\n[ERROR] You need to set your private key first!")
    print("\n  1. Open the .env file in this folder")
    print("  2. Replace 0xYOUR_PRIVATE_KEY_HERE with your actual private key")
    print("  3. Your private key is 64 hex chars (starts with 0x, 66 chars total)")
    print("  4. NOT your wallet address (that's only 42 chars)")
    print("\n  Example:  POLYMARKET_PRIVATE_KEY=0x1a2b3c4d...  (64 hex chars after 0x)")
    print("\n  MetaMask: Click account > Account Details > Export Private Key")
    sys.exit(1)

key_hex = private_key.replace("0x", "")
if len(key_hex) != 64:
    print(f"\n[ERROR] Your key is {len(key_hex)} hex chars — private keys must be exactly 64.")
    if len(key_hex) == 40:
        print("  That looks like a wallet ADDRESS, not a private key!")
        print("  MetaMask: Click account > Account Details > Export Private Key")
    sys.exit(1)

print("\nConnecting to Polymarket CLOB...")

try:
    from py_clob_client.client import ClobClient

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
    )

    print("Deriving API credentials...\n")
    credentials = client.create_or_derive_api_creds()

    print("=" * 50)
    print("  YOUR POLYMARKET API CREDENTIALS")
    print("=" * 50)
    print(f"  API Key:     {credentials.api_key}")
    print(f"  Secret:      {credentials.api_secret}")
    print(f"  Passphrase:  {credentials.api_passphrase}")
    print("=" * 50)
    print("\nThese are now active. The bot uses them automatically")
    print("from your private key — no need to paste them anywhere.")

except Exception as e:
    print(f"\n[ERROR] Failed to get credentials: {e}")
    print("\nCommon fixes:")
    print("  - Make sure your private key is correct")
    print("  - Make sure you have internet connection")
    print("  - Make sure py-clob-client is installed: pip install py-clob-client")
    sys.exit(1)
