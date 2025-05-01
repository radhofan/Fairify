import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load teacher model
teacher = load_model('Fairify/models/german/GC-2.h5')
teacher.trainable = False  # freeze teacher
print(teacher.summary())

# Load synthetic dataset
df = pd.read_csv('Fairify/experimentData/synthetic-german-predicted-gpt2.csv')
X = df.drop(columns=['credit'])
y = df['credit']

# Encode categorical features
categorical_columns = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors',
                       'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
for column in categorical_columns:
    label_encoder = LabelEncoder()
    X[column] = label_encoder.fit_transform(X[column])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Ensure y_train has the correct shape (1D array)
y_train = y_train.values.reshape(-1, 1)  # Convert to NumPy array and reshape

# Precompute teacher logits for training data (soft labels)
teacher_probs = teacher.predict(X_train)
teacher_logits = tf.math.log(teacher_probs + 1e-7) / 3.0  # Apply temperature scaling

# Clone model structure for student
student = clone_model(teacher)
student.build(input_shape=(None, X_train.shape[1]))

# Custom knowledge distillation loss
alpha = 0.7  # weight for soft loss
temperature = 3.0  # temperature for softening logits

def kd_loss(y_true, y_pred):
    # Binary crossentropy loss for hard targets
    ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Apply temperature scaling for the student logits
    student_logits = tf.math.log(y_pred + 1e-7) / temperature

    # KL divergence between soft labels (teacher logits) and student logits
    kl_loss = tf.keras.losses.KLDivergence()(teacher_logits, student_logits) * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss

# Compile student model
optimizer = Adam(learning_rate=0.0005)
student.compile(optimizer=optimizer, loss=kd_loss, metrics=['accuracy'])

# Train student model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
student.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save distilled student model
student.save('Fairify/models/german/GC-8-distilled.h5')
print("Distilled model saved as GC-8-distilled.h5")
