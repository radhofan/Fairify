# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# # Load pre-trained adult model
# model = load_model('Fairify/models/adult/AC-1.h5')
# print(model.summary())

# # Load synthetic data (mimicking GPT2-generated format)
# df = pd.read_csv('Fairify/experimentData/counterexamples.csv')

# # === Start of full preprocessing matching load_adult_ac1() ===

# # Drop rows with missing values
# df.dropna(inplace=True)

# # Define categorical columns to label encode
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# # Apply LabelEncoder
# for feature in cat_feat:
#     le = LabelEncoder()
#     df[feature] = le.fit_transform(df[feature])

# # Encode 'race' separately
# le_race = LabelEncoder()
# df['race'] = le_race.fit_transform(df['race'])

# # Bin capital-gain and capital-loss
# binning_cols = ['capital-gain', 'capital-loss']
# for feature in binning_cols:
#     bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#     df[feature] = bins.fit_transform(df[[feature]])

# df.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# # Split features and labels
# X = df.drop(columns=[label_name])
# y = df[label_name]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# # === End of preprocessing matching load_adult_ac1() ===

# # Compile model
# optimizer = Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train the model
# model.fit(
#     X_train, y_train,
#     epochs=100, 
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     callbacks=[early_stopping]
# )

# # Save retrained model
# model.save('Fairify/models/adult/AC-14.h5')
# print("Model retrained and saved as AC-14.h5")


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# Load pre-trained adult model
model = load_model('Fairify/models/adult/AC-1.h5')
print(model.summary())

# Load synthetic data (mimicking GPT2-generated format)
df = pd.read_csv('Fairify/experimentData/counterexamples.csv')

# === Start of full preprocessing matching load_adult_ac1() ===
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
y = df[label_name].values

# === End of preprocessing matching load_adult_ac1() ===

# -----------------------------
# Step 2: Relabel Using Label Propagation on CE Pairs
# -----------------------------

# Assume each pair of rows are CEs (i.e., even index = original, odd = CE)
y_relabels = []

for i in range(0, len(y), 2):
    if i + 1 >= len(y):  # skip incomplete pair
        break
    y_pair = y[i:i+2]
    avg = (y_pair[0] + y_pair[1]) / 2
    relabeled = 1 if avg >= 0.5 else 0
    y_relabels.extend([relabeled, relabeled])

y_relabels = np.array(y_relabels)
X_ce_pairs = X.iloc[:len(y_relabels)]

print(f"Relabeled {len(y_relabels)} instances across {len(y_relabels)//2} CE pairs")

# Train-test split with CE pairs
X_train, X_test, y_train, y_test = train_test_split(
    X_ce_pairs, y_relabels, test_size=0.15, random_state=42
)

# -----------------------------
# Step 3: Custom Fair Loss Function
# -----------------------------

lambda_fair = 0.5

class CustomFairLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_fair=0.5, name="custom_fair_loss"):
        super().__init__(name=name)
        self.lambda_fair = lambda_fair
    
    def call(self, y_true, y_pred):
        # Standard binary crossentropy loss
        task_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        task_loss = tf.reduce_mean(task_loss)
        
        # Fairness loss - minimize prediction differences between CE pairs
        # Ensure we have even number of samples for pairing
        batch_size = tf.shape(y_pred)[0]
        even_batch_size = batch_size - (batch_size % 2)
        
        y_pred_pairs = y_pred[:even_batch_size]
        y_pred_pairs = tf.reshape(y_pred_pairs, (-1, 2))
        
        # Calculate absolute difference between CE pairs
        fair_loss = tf.reduce_mean(tf.abs(y_pred_pairs[:, 0] - y_pred_pairs[:, 1]))
        
        return task_loss + self.lambda_fair * fair_loss

# Custom data generator to maintain CE pair structure
class CEPairGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        
        # Ensure batch size is even to maintain CE pairs
        if self.batch_size % 2 != 0:
            self.batch_size += 1
            
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X.iloc[batch_indices].values
        y_batch = self.y[batch_indices]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            # Shuffle while keeping CE pairs together
            pair_indices = np.arange(0, len(self.indices), 2)
            np.random.shuffle(pair_indices)
            new_indices = []
            for pair_start in pair_indices:
                if pair_start + 1 < len(self.indices):
                    new_indices.extend([pair_start, pair_start + 1])
            self.indices = np.array(new_indices)

# Compile model with custom fair loss
optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer, 
    loss=CustomFairLoss(lambda_fair=lambda_fair), 
    metrics=['accuracy']
)

# Create data generators
train_generator = CEPairGenerator(X_train, y_train, batch_size=32, shuffle=False)
val_generator = CEPairGenerator(X_test, y_test, batch_size=32, shuffle=False)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with fairness constraints
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the final model
test_loss, test_acc = model.evaluate(val_generator, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss (with fairness): {test_loss:.4f}")

# Save retrained model
model.save('Fairify/models/adult/AC-14.h5')
print("Model retrained with fairness constraints and saved as AC-14.h5")