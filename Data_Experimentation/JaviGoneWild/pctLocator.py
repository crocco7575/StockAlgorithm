with open('pctdictionary.txt', 'r') as filehandle:
    pctdistro_dictionary = [line.strip() for line in filehandle]

sum = 0
for ratings in pctdistro_dictionary:
    rating_list = str(ratings).split(",")
    volatility_rating = int(rating_list[0])
    price_rating = int(rating_list[1])
    pct = float(rating_list[2])

    sum += pct
    if sum > 0.8202:
        print(f"Vol: {volatility_rating}, Price: {price_rating}")
        break


