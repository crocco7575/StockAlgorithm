

our_stock_dict= {'key1': 3.5, 'key2': 1.2, 'key3': 2.8, 'key4': 0.5}
print(f"old: {our_stock_dict}")
def update_add_to_dict():
    global our_stock_dict
    new_stock_dict = our_stock_dict.copy()
    input_price = 0
    current_pos_list = 'key1', 'key2', 'key6', 'key7'
    for symbol in current_pos_list:
        if symbol in new_stock_dict:
            input_price = update_stock_price(symbol, new_stock_dict[symbol])
            if input_price < 0:
                new_stock_dict.pop(symbol)
            else:
                new_stock_dict[symbol] = input_price
        else:
            new_stock_dict[symbol] = 69

    print(f"in function{new_stock_dict}")
    our_stock_dict = new_stock_dict


def update_stock_price(symbol, price):
    cp = 15
    pp = float(price)
    new_price = 0
    if cp >= pp:
        new_price = cp
    else:
        if cp < (pp-(pp*trailing_stop_loss_constant)):
            new_price = -1
        else:
            new_price = pp
    return new_price
update_add_to_dict()
print(f"reassigned: {our_stock_dict}")