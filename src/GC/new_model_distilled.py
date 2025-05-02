# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model, clone_model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# # Load teacher model
# teacher = load_model('Fairify/models/german/GC-2.h5')
# teacher.trainable = False
# print(teacher.summary())

# # Load synthetic dataset
# df = pd.read_csv('Fairify/experimentData/synthetic-german-predicted-gpt2.csv')
# X = df.drop(columns=['credit'])
# y = df['credit']

# # Encode categorical features
# categorical_columns = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors',
#                        'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
# for column in categorical_columns:
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# # Precompute teacher soft labels (logits with temperature)
# temperature = 3.0
# teacher_probs = teacher.predict(X_train)
# teacher_soft = np.log(teacher_probs + 1e-7) / temperature  # shape: (n_samples, 1)

# # Clone student model
# student = clone_model(teacher)
# student.build(input_shape=(None, X_train.shape[1]))

# # Custom knowledge distillation loss
# alpha = 0.7

# def kd_loss(y_true, y_pred):
#     y_true, teacher_soft = y_true[:, 0:1], y_true[:, 1:2]  # extract both
#     ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     student_log = tf.math.log(y_pred + 1e-7) / temperature
#     kl = tf.keras.losses.KLDivergence()(teacher_soft, student_log) * (temperature ** 2)
#     return alpha * ce + (1 - alpha) * kl

# # Combine hard and soft labels into one array
# y_train_combined = np.hstack([np.array(y_train).reshape(-1, 1), teacher_soft])

# # Create tf.data.Dataset
# train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train_combined)).batch(32).shuffle(512)
# val_ds = tf.data.Dataset.from_tensor_slices((X_test.values, np.array(y_test).reshape(-1, 1))).batch(32)

# # Compile and train
# student.compile(optimizer=Adam(0.0005), loss=kd_loss, metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# student.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping])

# # Save model
# student.save('Fairify/models/german/GC-8.h5')
# print("✅ Distilled model saved as GC-8.h5")



import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Add gradient clipping
tf.keras.backend.clear_session()
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Add gradient clipping

# Load teacher model
teacher = load_model('Fairify/models/german/GC-2.h5')
teacher.trainable = False
print(teacher.summary())

# Load synthetic dataset
df = pd.read_csv('Fairify/experimentData/synthetic-german-predicted-gpt2.csv')
X = df.drop(columns=['credit'])
y = df['credit']

# Encode categorical features
categorical_columns = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors',
                       'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Normalize numerical features to prevent extreme values
numerical_columns = [col for col in X.columns if col not in categorical_columns]
if numerical_columns:
    for col in numerical_columns:
        X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-7)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Generate soft labels from teacher model with proper temperature scaling
temperature = 2.0  # Reduced from 3.0 to prevent extreme values
teacher_probs_raw = teacher.predict(X_train)

# For binary classification, create both class probabilities
teacher_probs = np.column_stack([1-teacher_probs_raw, teacher_probs_raw])

# Apply temperature scaling to soften the probabilities
teacher_logits = np.log(np.clip(teacher_probs, 1e-7, 1-1e-7))
teacher_logits_T = teacher_logits / temperature
teacher_probs_T = tf.nn.softmax(teacher_logits_T, axis=1).numpy()

# Store original hard labels for later use
y_train_hard = y_train

# Custom loss function with improved numerical stability
def distillation_loss(y_true, y_pred):
    # Extract components from y_true
    hard_labels = tf.cast(y_true[:, 0:1], tf.float32)  # Original labels
    soft_labels_0 = tf.cast(y_true[:, 1:2], tf.float32)  # Teacher's prob for class 0
    soft_labels_1 = tf.cast(y_true[:, 2:3], tf.float32)  # Teacher's prob for class 1
    soft_labels = tf.concat([soft_labels_0, soft_labels_1], axis=1)
    
    # Reshape y_pred for binary classification
    y_pred_reshaped = tf.concat([1-y_pred, y_pred], axis=1)
    
    # Standard binary cross-entropy with hard labels
    hard_loss = tf.keras.losses.binary_crossentropy(hard_labels, y_pred)
    
    # KL divergence with soft labels (with numerical stability)
    y_pred_safe = tf.clip_by_value(y_pred_reshaped, 1e-7, 1-1e-7)
    soft_loss = tf.reduce_sum(soft_labels * tf.math.log(soft_labels / y_pred_safe + 1e-7), axis=1)
    
    # Apply temperature scaling factor
    soft_loss = soft_loss * (temperature ** 2)
    
    # Balance between hard and soft losses
    alpha = 0.5  # Modified from 0.7 to give more weight to soft labels
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss

# Prepare data for training with soft labels
y_train_combined = np.column_stack([
    y_train_hard,  # Original hard labels
    teacher_probs_T[:, 0],  # Soft label for class 0
    teacher_probs_T[:, 1]   # Soft label for class 1
])

# Create datasets with proper batch size
batch_size = 64  # Larger batch size for more stable gradients
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train_combined)).batch(batch_size).shuffle(1024)

# For validation, we need a compatible format but will only use hard labels for metrics
dummy_soft_labels = np.zeros((len(y_test), 2))  # Doesn't matter for validation metrics
y_val_combined = np.column_stack([y_test, dummy_soft_labels])
val_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_val_combined)).batch(batch_size)

# Create student model with proper initialization 
student = clone_model(teacher)
student.build(input_shape=(None, X_train.shape[1]))

# Add debugging metrics
def accuracy(y_true, y_pred):
    # Only use the hard labels for accuracy
    hard_labels = tf.cast(y_true[:, 0:1], tf.float32)
    return tf.keras.metrics.binary_accuracy(hard_labels, y_pred)

# Compile with improved settings
student.compile(
    optimizer=optimizer,
    loss=distillation_loss,
    metrics=[accuracy],
    run_eagerly=True  # For debugging
)

# Add callbacks for better training stability
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
]

# Train with verbose output to monitor progress
history = student.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Reduced from 100
    callbacks=callbacks,
    verbose=2
)

# Validate performance before saving
test_loss, test_acc = student.evaluate(val_ds)
print(f"Test accuracy: {test_acc:.4f}")

# Save model
student.save('Fairify/models/german/GC-8.h5')
print("✅ Distilled model saved as GC-8.h5")