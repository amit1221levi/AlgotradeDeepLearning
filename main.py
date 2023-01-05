import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Load stock data

df = pd.read_csv("stock_data.csv")

#Calculate volatility

df['log_return'] = np.log(df['close']).diff()
df['volatility'] = df['log_return'].rolling(window=252).std() * np.sqrt(252)

#Scale data

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['volatility']])

#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(scaled_df, df['label'], test_size=0.3)

#Build deep learning model

model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#Compile and fit model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)

#Evaluate model

score = model.evaluate(X_test, y_test, batch_size=32)
print("Loss: ", score[0])
print("Accuracy: ", score[1])

#Use model to make predictions on test data

predictions = model.predict(X_test)
predictions = [1 if x > 0.5 else 0 for x in predictions]

#Calculate accuracy of predictions

accuracy = sum([1 if predictions[i] == y_test.iloc[i] else 0 for i in range(len(predictions))]) / len(predictions)
print("Prediction Accuracy: ", accuracy)