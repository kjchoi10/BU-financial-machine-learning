# Boston Univeristy Financial Machine Learning Research
Predicting the S&amp;P 500 stock market prices using machine learning algorithms and techniques.

by Rashi Verma, Zoey Zhang, Giang Qiao, and Kevin Choi
with faculty advisor Mark Kon


# Primary Goal: 

To set up a Random Forest Machine that trains on monthly stock data and predicts the returns over a testing period. The predicted portfolio of the top companies is dynamically updated for each test month, and the investment of a dollar at the beginning of the testing period is tested against the S&P500 index.

# Secondary Goal:
To prepare a list of technicals (included in the appendix as separate functions), and rank them according to performance.

# Procedure:
The following blocks deal with each of these steps in a separate subroutine:

• The first step (not included here) is to import monthly price data from Yahoo (done using an R code, courtesy Ms. Zoey Zheng).

• Reading the csv data file of monthly returns

• Assigning training as well as testing periods

• Building up training and testing matrices based on chosen technicals

• Generating the relevant plots.
