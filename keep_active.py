import requests
import time

while True:
    try:
        requests.get("https://your-app.streamlit.app")
        print("Ping sent.")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(36000)  # 10 hours