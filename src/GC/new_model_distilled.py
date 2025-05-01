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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Precompute teacher soft labels with temperature
temperature = 3.0
teacher_probs = teacher.predict(X_train)
teacher_soft = np.log(np.vstack([1 - teacher_probs[:, 0], teacher_probs[:, 0]]).T) / temperature

# Clone student model
student = clone_model(teacher)
student.build(input_shape=(None, X_train.shape[1]))

# Custom knowledge distillation loss for binary classification
alpha = 0.7

def kd_loss(y_true, y_pred):
    # Split the combined target
    true_labels = y_true[:, 0:1]  # True binary labels
    teacher_logits = y_true[:, 1:]  # Teacher soft labels
    
    # Standard binary cross-entropy loss
    ce_loss = tf.keras.losses.binary_crossentropy(true_labels, y_pred)
    
    # Distillation loss (KL divergence between teacher and student logits)
    student_logits = tf.math.log(tf.concat([1 - y_pred, y_pred], axis=1)) / temperature
    kl_loss = tf.reduce_mean(
        tf.keras.losses.kl_divergence(teacher_logits, student_logits)) * (temperature ** 2)
    
    return alpha * ce_loss + (1 - alpha) * kl_loss

# Combine hard and soft labels
y_train_combined = np.hstack([
    np.array(y_train).reshape(-1, 1),  # True labels
    teacher_soft  # Teacher soft labels
])

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train_combined)).batch(32).shuffle(512)
val_ds = tf.data.Dataset.from_tensor_slices((X_test.values, np.array(y_test).reshape(-1, 1))).batch(32)

# Compile and train
student.compile(optimizer=Adam(0.0005), 
               loss=kd_loss, 
               metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = student.fit(train_ds, 
                     validation_data=val_ds, 
                     epochs=100, 
                     callbacks=[early_stopping])

# Save model
student.save('Fairify/models/german/GC-8.h5')
print("✅ Distilled model saved as GC-8.h5")