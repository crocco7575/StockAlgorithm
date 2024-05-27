import schedule
import requests
import time


def run_main_logic():

    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKJIDRFO8XYMG1VJ2HZB",
        "APCA-API-SECRET-KEY": "tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3"
    }

    response = requests.delete(url, headers=headers)

    print(response.text)


schedule.every().day.at("15:55").do(run_main_logic)

while True:
    schedule.run_pending()
    time.sleep(1)
