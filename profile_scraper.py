import json
import rsa
import requests
import base64

from bs4 import BeautifulSoup

from icecream import ic
rsa_server = (
    'https://api.steampowered.com/'
    'IAuthenticationService/GetPasswordRSAPublicKey/v1/'
)
usr_name = ''
response = requests.get(rsa_server, params={'account_name': 'usr_name'})
rsa_key_data = response.json()['response']

public_key_encoded = rsa_key_data['publickey_mod'], rsa_key_data['publickey_exp']
encryption_timestamp = rsa_key_data['timestamp']

modulus = int(public_key_encoded[0], 16)
exponent = int(public_key_encoded[1], 16)

public_key = rsa.PublicKey(modulus, exponent)

# Encrypt your password
password = ''
encrypted_password = rsa.encrypt(password.encode('utf-8'), public_key)

# Base64 encode the encrypted password
encrypted_password_b64 = base64.b64encode(encrypted_password)

print(encrypted_password_b64)
ic(encryption_timestamp)