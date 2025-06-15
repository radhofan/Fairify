import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# Load pre-trained adult model
model = load_model('Fairify/models/adult/AC-1.h5')
print(model.summary())

# Load synthetic data (mimicking GPT2-generated format)
df = pd.read_csv('Fairify/experimentData/counterexamples-fair-lfr.csv')

# === Start of full preprocessing matching load_adult_ac1() ===

# Drop unnecessary columns
del_cols = ['fnlwgt']
df.drop(labels=del_cols, axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Define categorical columns to label encode
cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'native-country', 'sex']

# Apply LabelEncoder
for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

# Encode 'race' separately
le_race = LabelEncoder()
df['race'] = le_race.fit_transform(df['race'])

# Bin capital-gain and capital-loss
binning_cols = ['capital-gain', 'capital-loss']
for feature in binning_cols:
    bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    df[feature] = bins.fit_transform(df[[feature]])

df.rename(columns={'decision': 'income-per-year'}, inplace=True)
label_name = 'income-per-year'

# Split features and labels
X = df.drop(columns=[label_name])
y = df[label_name]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# === End of preprocessing matching load_adult_ac1() ===

# Compile model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    epochs=100, 
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save retrained model
model.save('Fairify/models/adult/AC-14.h5')
print("Model retrained and saved as AC-14.h5")