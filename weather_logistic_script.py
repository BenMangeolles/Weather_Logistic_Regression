import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
#--------------- PULL IN HISTORICAL DATA ------------------------------

def historical_weather_data():
    historical_weather_NY = pd.read_csv("NY_Historical_Weather_Data.csv")
    print(historical_weather_NY)
    return historical_weather_NY

def historical_DJX_data():
    historical_DJX = pd.read_csv("DJA_Price_History.csv")
    print(historical_DJX)
    return historical_DJX

def historical_df_merge(DJX_df, weather_df):
    merged_df = pd.merge(DJX_df, weather_df, on='date', how='inner')
    merged_df['Close'] = merged_df['Close'].astype(float)
    print(merged_df)

    return merged_df

#------------------------------------ Load data -------------------------------
DJX_df = historical_DJX_data()
weather_df = historical_weather_data()

#Merge data
merged_df = historical_df_merge(DJX_df, weather_df)

#Create a target variable where 1 indicates the close price was higher than the open price
merged_df['target'] = (merged_df['Close'] > merged_df['Open']).astype(int)

print(merged_df)

#--------------------------- SCRAPE WEATHER DATA FOR TODAYS PRICE PREDICTION -----------------------
def scrape_weather_data():
    weather_website_url = 'https://www.bbc.co.uk/weather/5128581'
    response = requests.get(weather_website_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    #Extract temperature
    curr_temp = soup.find('span', class_='wr-value--temperature--c').text
    curr_temp = int(curr_temp.replace('°', ''))  #Remove the degree symbol and convert to integer

    #Extract rain chance
    curr_precipprob = soup.find('span', class_='wr-u-font-weight-500').text
    curr_precipprob = int(curr_precipprob.replace('%', ''))  #Remove the % symbol and convert to integer

    curr_windspeed = soup.find('span', class_= "wr-value--windspeed wr-value--windspeed--kph").text
    curr_windspeed = int(curr_windspeed.replace(' km/h', ''))

    return pd.DataFrame({
        'temp': [curr_temp],
        'precipprob': [curr_precipprob],
        'windspeed': [curr_windspeed]
    })

#Call the function and print the results

print(scrape_weather_data()) #Check that everything pulled in correctly


#------------------- LOGISTIC REGRESSION MODEL TRAINING WORK ----------------------------------

def main():
    data = merged_df

    #Add constant to feature set
    X = data[['temp', 'precipprob', 'windspeed']]
    X = sm.add_constant(X)  # adding a constant

    y = data['target']

    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#Edit size and random_state params, used 20% as industry standard

    #Fit the logistic regression model using statsmodels
    model = sm.Logit(y_train, X_train)
    result = model.fit()


    print(result.summary())

    # redict using the model

    y_pred = result.predict(sm.add_constant(X_test)) #Adding constant term as statsmodel doesnt for regression models for some reason
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]  #Convert probabilities to class labels
    print("Predictions:", y_pred)


    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
