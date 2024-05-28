import schedule
import pytz
from datetime import datetime, time
import time as t


def main_print():
    watchlist = []
    with open('watchlist.txt', 'r') as filehandle:  # grabs watchlist
        for line in filehandle:
            watch_ticker = line[:-1]
            watchlist.append(watch_ticker)
    for stock in watchlist:
        print(stock)
    target_timezone_1 = pytz.timezone('US/Eastern')
    current_time_1 = datetime.now(target_timezone_1).time()
    print(current_time_1)


schedule.every().day.at("10:10").do(main_print)
schedule.every().day.at("15:11").do(schedule.clear)

target_time = time(10, 10)
target_timezone = pytz.timezone('US/Eastern')

first_iteration = True

while True:
    # Get the current time in EST
    current_time = datetime.now(target_timezone).time()

    # Get the current datetime in EST
    current_datetime = datetime.now(target_timezone).replace(microsecond=0, second=0)

    # Combine the current date with the target time
    target_datetime = current_datetime.replace(hour=target_time.hour, minute=target_time.minute)

    # Check if the current time is equal to the target time
    if current_datetime.time() == target_time:
        if first_iteration:
            print("Intiating 1 min schedule...")
            schedule.every(1).minutes.do(main_print)
            first_iteration = False
# Keep the script running
    schedule.run_pending()
    t.sleep(1)