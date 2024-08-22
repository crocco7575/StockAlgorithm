from collections import Counter

with open('pricevoldistrolist.txt', 'r') as filehandle:
    rating_dictionary = [line.strip() for line in filehandle]

stripped_ratings = []
for stock in rating_dictionary:
    new_rating = str(stock).split(",")
    stripped_ratings.append(new_rating[1] + "," + new_rating[2])

count = Counter(stripped_ratings)

final_pctg_list = []
for string, freq in count.items():
    final_pct = freq / 3813
    print(f"Ratings: {string}, Freq: {freq}, Pct: {final_pct}")
    final_pctg_list.append(string + "," + str(final_pct))

with open('pctdictionary.txt', 'w') as file:  # updates with new stock
    for obj in final_pctg_list:
        file.write(obj + "\n")

