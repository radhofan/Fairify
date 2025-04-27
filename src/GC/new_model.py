import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

model = load_model('Fairify/models/german/GC-2.h5')

# One
# df = pd.read_csv('Fairify/experimentData/synthetic-german-one-ctgan.csv') # sdg ctgan
# df = pd.read_csv('Fairify/experimentData/synthetic-german-one-distilgpt2.csv') # great distill gpt2
# df = pd.read_csv('Fairify/experimentData/synthetic-german-one-gpt2.csv') # great gpt2

# Predicted
# df = pd.read_csv('Fairify/experimentData/synthetic-german-predicted-distilgpt2.csv') # great distill gpt2
df = pd.read_csv('Fairify/experimentData/synthetic-german-predicted-gpt2.csv') # great gpt2

X = df.drop(columns=['credit'])  
y = df['credit']

categorical_columns = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
for column in categorical_columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# 0.001 - jlek
# 0.01 - kek taik
# 0.0001 - 
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='hinge', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# for sdg ctgan
# model.save('Fairify/models/german/GC-6.h5')
# print("Model retrained and saved as GC-6.h5")

# for distill gpt2
# model.save('Fairify/models/german/GC-7.h5')
# print("Model retrained and saved as GC-7.h5")

# for gpt2
model.save('Fairify/models/german/GC-8.h5')
print("Model retrained and saved as GC-8.h5")
