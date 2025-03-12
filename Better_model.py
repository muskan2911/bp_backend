import tensorflow as tf
import numpy as np
import h5py
import os
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from scipy import signal

# Path where you want to save the model
MODEL_PATH = './improved_bp_resnet_model'

def load_dataset(filename, num_samples=None):
    with h5py.File(filename, 'r') as h5:
        ppg = h5['ppg'][:num_samples]
        labels = h5['label'][:num_samples]
        subject_idx = h5['subject_idx'][:num_samples]
    return ppg, labels, subject_idx

def preprocess_data(ppg, labels):
    # Z-score normalization (typically better than min-max for physiological signals)
    ppg = (ppg - np.mean(ppg, axis=1, keepdims=True)) / (np.std(ppg, axis=1, keepdims=True) + 1e-10)
    
    # Apply bandpass filter to remove noise (0.5-8Hz is typical for PPG)
    fs = 125  # Assuming 125Hz sampling rate, adjust if different
    sos = signal.butter(4, [0.5, 8], 'bandpass', fs=fs, output='sos')
    
    # Apply filter to each signal
    filtered_ppg = np.zeros_like(ppg)
    for i in range(ppg.shape[0]):
        filtered_ppg[i] = signal.sosfilt(sos, ppg[i])
    
    # Add channel dimension
    filtered_ppg = filtered_ppg[..., np.newaxis]
    
    # Extract SBP and DBP
    sbp = labels[:, 0]
    dbp = labels[:, 1]
    
    return filtered_ppg, sbp, dbp

def data_augmentation(ppg, sbp, dbp, augment_ratio=0.3):
    """Simple time-domain augmentation for PPG signals"""
    n_samples = int(augment_ratio * len(ppg))
    indices = np.random.choice(len(ppg), n_samples, replace=False)
    
    aug_ppg = []
    aug_sbp = []
    aug_dbp = []
    
    for idx in indices:
        # Time shifting
        shift = np.random.randint(-50, 50)
        shifted_ppg = np.roll(ppg[idx], shift, axis=0)
        aug_ppg.append(shifted_ppg)
        aug_sbp.append(sbp[idx])
        aug_dbp.append(dbp[idx])
        
        # Amplitude scaling (small variations)
        scale = np.random.uniform(0.9, 1.1)
        scaled_ppg = ppg[idx] * scale
        aug_ppg.append(scaled_ppg)
        aug_sbp.append(sbp[idx])
        aug_dbp.append(dbp[idx])
    
    # Combine original and augmented data
    ppg_combined = np.concatenate([ppg, np.array(aug_ppg)], axis=0)
    sbp_combined = np.concatenate([sbp, np.array(aug_sbp)], axis=0)
    dbp_combined = np.concatenate([dbp, np.array(aug_dbp)], axis=0)
    
    return ppg_combined, sbp_combined, dbp_combined

def resnet_block(input_tensor, filters, kernel_size=3, strides=1, use_batch_norm=True, dropout_rate=0.2):
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Add dropout for regularization
    x = Dropout(dropout_rate)(x)
    
    x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer="he_normal")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    
    # Skip connection with appropriate dimensions
    if strides != 1 or input_tensor.shape[-1] != filters:
        input_tensor = Conv1D(filters, kernel_size=1, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
        if use_batch_norm:
            input_tensor = BatchNormalization()(input_tensor)
    
    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x

def ImprovedResNet1D(input_shape, num_blocks, filters, dropout_rate=0.2):
    X_input = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv1D(filters[0], kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(X_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # ResNet blocks
    for i, f in enumerate(filters):
        for j in range(num_blocks[i]):
            strides = 2 if j == 0 and i > 0 else 1
            x = resnet_block(x, f, strides=strides, dropout_rate=dropout_rate)
    
    # Global pooling and prediction layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    
    # Deeper prediction branches for each output
    sbp_branch = Dense(128, activation="relu")(x)
    sbp_branch = Dropout(dropout_rate)(sbp_branch)
    sbp_branch = Dense(64, activation="relu")(sbp_branch)
    SBP = Dense(1, activation="linear", name="SBP")(sbp_branch)
    
    dbp_branch = Dense(128, activation="relu")(x)
    dbp_branch = Dropout(dropout_rate)(dbp_branch)
    dbp_branch = Dense(64, activation="relu")(dbp_branch)
    DBP = Dense(1, activation="linear", name="DBP")(dbp_branch)
    
    model = Model(inputs=X_input, outputs=[SBP, DBP], name="ImprovedResNet1D")
    return model

def save_trained_model():
    # Load and preprocess the data
    filename = "./Mimic.h5" 
    ppg, labels, subject_idx = load_dataset(filename, num_samples=None)  # Use all available data
    ppg, sbp, dbp = preprocess_data(ppg, labels)
    
    # Split the data - use subject-wise splitting if possible
    unique_subjects = np.unique(subject_idx)
    n_subjects = len(unique_subjects)
    val_subjects = unique_subjects[int(0.8 * n_subjects):]
    
    val_mask = np.isin(subject_idx, val_subjects)
    train_mask = ~val_mask
    
    ppg_train, sbp_train, dbp_train = ppg[train_mask], sbp[train_mask], dbp[train_mask]
    ppg_val, sbp_val, dbp_val = ppg[val_mask], sbp[val_mask], dbp[val_mask]
    
    # Augment training data
    ppg_train, sbp_train, dbp_train = data_augmentation(ppg_train, sbp_train, dbp_train)
    
    # Define and compile the improved model
    input_shape = ppg.shape[1:]  # Should be (875, 1)
    num_blocks = [3, 4, 6]  # Deeper architecture
    filters = [64, 128, 256, 512]  # More filters
    
    model = ImprovedResNet1D(input_shape, num_blocks, filters)
    
    # Use Huber loss which is less sensitive to outliers than MSE
    huber_loss = Huber(delta=1.0)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"SBP": huber_loss, "DBP": huber_loss},
        metrics={"SBP": "mae", "DBP": "mae"}
    )
    
    # Callbacks for better training
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_SBP_mae',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_SBP_mae',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_PATH, 'best_model.h5'),
        monitor='val_SBP_mae',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model with callbacks
    history = model.fit(
        ppg_train,
        {"SBP": sbp_train, "DBP": dbp_train},
        validation_data=(ppg_val, {"SBP": sbp_val, "DBP": dbp_val}),
        epochs=50,  # Increase epochs but use early stopping
        batch_size=32,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint]
    )
    
    # Save the final model
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model.save(os.path.join(MODEL_PATH, 'final_model'))
    
    # Evaluate the model
    evaluation = model.evaluate(
        ppg_val, {"SBP": sbp_val, "DBP": dbp_val}
    )
    
    print(f"Final validation SBP MAE: {evaluation[3]}")
    print(f"Final validation DBP MAE: {evaluation[4]}")
    
    return model, history

if __name__ == "__main__":
    if not os.path.exists(os.path.join(MODEL_PATH, 'final_model')):
        print("Training and saving improved model...")
        model, history = save_trained_model()
    else:
        print(f"Model already exists at {MODEL_PATH}")
        # Load the model if needed
        model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'final_model'))