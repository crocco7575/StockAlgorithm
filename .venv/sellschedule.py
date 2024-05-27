import schedule
import subprocess
import alpaca_trade_api as alp

print("Sell schedule activated")
api = alp.REST('PKJIDRFO8XYMG1VJ2HZB',
               'tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3',
               'https://paper-api.alpaca.markets'
               )

if api.get_clock().is_open is False:
    exit()

subprocess.run(['python', 'sellCode.py'])
def run_another_script():
    subprocess.run(['python', 'sellCode.py'])


interval = 5

schedule.every(interval).minutes.do(run_another_script)

# Run the scheduled tasks indefinitely
while True:
    # Run pending scheduled tasks
    schedule.run_pending()
