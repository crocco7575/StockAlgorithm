import schedule
import alpaca_trade_api as alp



def job():
    print("yessir")

    api = alp.REST('PKJIDRFO8XYMG1VJ2HZB',
                'tK0FDThUKft0vfC2NBmmo8BEQC7atyNgpkiFy5k3',
                'https://paper-api.alpaca.markets'
                   )

    #if api.get_clock().is_open is False:
        #exit()

    with open('daily_segmentation_number.txt', 'w') as filehandle:
        filehandle.write('19')

    interval = 20

    schedule.every(interval).minutes.do(run_another_script)

    # Run the scheduled tasks indefinitely
    while True:
        # Run pending scheduled tasks
        schedule.run_pending()

def run_another_script():
    #subtract one from segmentation number
    new_number = 0
    with open('daily_segmentation_number.txt', 'r') as filehandle:
        for line in filehandle:
            new_number = int(line) - 1

    with open('daily_segmentation_number.txt', 'w') as filehandle:
        filehandle.write(str(new_number))

    if new_number == 0:
        exit()

    if new_number > 0:
        subprocess.run(['python', 'buyCode.py'])



schedule.every().day.at("22:05").do(job)
# Run the scheduled tasks indefinitely
while True:
    # Run pending scheduled tasks
    schedule.run_pending()

