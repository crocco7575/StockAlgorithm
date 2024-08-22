from joblib import load
import pandas as pd
def make_prediction(flag_price, buy_price, time_between_flag_and_buy, lowest_rsi, ttm_strength):
	# Load the trained model
	clf = load('random_forest_model.joblib')

	# Create a dictionary to store the input parameters
	input_params = {
		'Flag Price': flag_price,
		'Buy Price': buy_price,
		'Time Between Flag and Buy': time_between_flag_and_buy,
		'Lowest RSI': lowest_rsi,
		'TTM Strength': ttm_strength
	}

	# Convert the dictionary to a pandas DataFrame
	input_df = pd.DataFrame(input_params, index=[0])

	# Make a prediction using the trained model
	prediction = clf.predict(input_df)

	# Return 'Good Trade' if the prediction is 1, otherwise return 'Bad Trade'
	return 'Good Trade' if prediction[0] == 1 else 'Bad Trade'

# Example usage:
flag_price = 11.45
buy_price = 11.72
time_between_flag_and_buy = 9
lowest_rsi = 24.5205513733068
ttm_strength = 0.090128580161504


result = make_prediction(flag_price, buy_price, time_between_flag_and_buy, lowest_rsi, ttm_strength)
print(result)  # Output: Good Trade or Bad Trade