import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

model = load_model('Fairify/models/bank/BM-1.h5')
print(model.summary())

df = pd.read_csv('Fairify/experimentData/synthetic-bank-predicted-gpt2.csv') 

X = df.drop(columns=['y'])  
y = df['y']

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'emp.var.rate', 'duration',
                        'campaign', 'pdays', 'previous', 'poutcome', 'age']
for column in categorical_columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# for gpt2
model.save('Fairify/models/bank/BM-6.h5')
print("Model retrained and saved as BM-6.h5")

