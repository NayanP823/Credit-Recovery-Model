import requests
import json

url = "http://127.0.0.1:8000/assess_risk"

# Sample data (based on a high risk profile potentially)
data = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 2,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 2,
    "PAY_4": 2,
    "PAY_5": 2,
    "PAY_6": 2,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 45000,
    "BILL_AMT3": 40000,
    "BILL_AMT4": 38000,
    "BILL_AMT5": 35000,
    "BILL_AMT6": 30000,
    "PAY_AMT1": 200,
    "PAY_AMT2": 200,
    "PAY_AMT3": 150,
    "PAY_AMT4": 150,
    "PAY_AMT5": 100,
    "PAY_AMT6": 100
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Connection failed: {e}")
