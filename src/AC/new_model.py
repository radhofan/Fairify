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
y_soft_labels = []

for i in range(0, len(y), 2):
    if i + 1 >= len(y):  # skip incomplete pair
        break
    y1, y2 = y[i], y[i+1]
    propagated = (y1 + y2) / 2  # harmonic propagation
    y_soft_labels.extend([propagated, propagated])  # assign same label to both

y_relabels = np.array(y_soft_labels)
X_ce_pairs = X.iloc[:len(y_soft_labels)]

print(f"Relabeled {len(y_relabels)} instances across {len(y_relabels)//2} CE pairs")

unique, counts = np.unique(y_relabels, return_counts=True)
print(dict(zip(unique, counts)))

# Train-test split with CE pairs
X_train, X_test, y_train, y_test = train_test_split(
    X_ce_pairs, y_relabels, test_size=0.15, random_state=42
)

# -----------------------------
# Step 3: Fair Training with Custom Training Loop
# -----------------------------

lambda_fair = 0.5

# Compile model with standard loss for now
optimizer = Adam(learning_rate=0.0001)  # Increase learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Add regularization to prevent overfitting
from tensorflow.keras import regularizers

# Get the current model architecture and add dropout/regularization
if hasattr(model, 'layers'):
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(0.01)

# Custom training function
def fair_train_step(model, X_batch, y_batch, lambda_fair=0.5):
    with tf.GradientTape() as tape:
        predictions = model(X_batch, training=True)
        
        # Reshape y_batch to match predictions shape
        y_batch = tf.reshape(y_batch, (-1, 1))
        
        # Standard loss
        task_loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)
        task_loss = tf.reduce_mean(task_loss)
        
        # Fairness loss - for CE pairs
        batch_size = tf.shape(predictions)[0]
        even_batch_size = batch_size - (batch_size % 2)
        
        if even_batch_size >= 2:
            pred_pairs = predictions[:even_batch_size]
            pred_pairs = tf.reshape(pred_pairs, (-1, 2))
            fair_loss = tf.reduce_mean(tf.abs(pred_pairs[:, 0] - pred_pairs[:, 1]))
            
            # Add small epsilon to prevent exact zero
            fair_loss = tf.maximum(fair_loss, 1e-8)
        else:
            fair_loss = 0.0
        
        # Reduce lambda_fair to prevent domination
        total_loss = task_loss + 0.05 * fair_loss  # Reduced from lambda_fair
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, task_loss, fair_loss

# Custom training loop
def train_fair_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    best_val_loss = float('inf')
    patience_count = 0
    patience = 3
    
    for epoch in range(epochs):
        # Training
        epoch_losses = []
        epoch_task_losses = []
        epoch_fair_losses = []
        
        # Shuffle training data while keeping pairs together
        n_pairs = len(X_train) // 2
        pair_indices = np.arange(n_pairs)
        np.random.shuffle(pair_indices)
        
        shuffled_indices = []
        for pair_idx in pair_indices:
            shuffled_indices.extend([pair_idx * 2, pair_idx * 2 + 1])
        
        X_train_shuffled = X_train.iloc[shuffled_indices]
        y_train_shuffled = y_train[shuffled_indices]
        
        # Batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled.iloc[i:i+batch_size].values
            y_batch = y_train_shuffled[i:i+batch_size]
            
            if len(X_batch) < 2:  # Skip small batches
                continue
                
            total_loss, task_loss, fair_loss = fair_train_step(
                model, X_batch, y_batch, lambda_fair
            )
            
            epoch_losses.append(total_loss.numpy())
            epoch_task_losses.append(task_loss.numpy())
            epoch_fair_losses.append(fair_loss.numpy() if isinstance(fair_loss, tf.Tensor) else fair_loss)
        
        # Validation
        val_loss = model.evaluate(X_val.values, y_val, verbose=0)[0]
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {np.mean(epoch_losses):.4f} - "
              f"Task Loss: {np.mean(epoch_task_losses):.4f} - "
              f"Fair Loss: {np.mean(epoch_fair_losses):.4f} - "
              f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            model.save_weights('best_weights.h5')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best weights
    model.load_weights('best_weights.h5')
    return model

# Train the model with fairness constraints
print("Starting fair training...")
model = train_fair_model(model, X_train, y_train, X_test, y_test)

# Final evaluation
test_loss, test_acc = model.evaluate(X_test.values, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# Save retrained model (now with standard loss, no custom objects)
model.save('Fairify/models/adult/AC-14.h5')
print("Model retrained with fairness constraints and saved as AC-14.h5")