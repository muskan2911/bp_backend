# save_model.py
import tensorflow as tf
import os

# Path where you want to save the model
MODEL_PATH = './bp_resnet_model'

# This assumes 'model' is your trained model from the provided code
# If the model variable is still in memory, you can directly use:
# model.save(MODEL_PATH)

# If you need to re-run your training code:
def save_trained_model():
    # Import your training code
    # This is a simplified version, you should modify to match your actual implementation
    import numpy as np
    import h5py
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
    from tensorflow.keras.models import Model
    from sklearn.model_selection import train_test_split
    
    def load_dataset(filename, num_samples=None):
        with h5py.File(filename, 'r') as h5:
            ppg = h5['ppg'][:num_samples]
            labels = h5['label'][:num_samples]
            subject_idx = h5['subject_idx'][:num_samples]
        return ppg, labels, subject_idx
    
    def preprocess_data(ppg, labels):
        ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
        ppg = ppg[..., np.newaxis]
        sbp = labels[:, 0]
        dbp = labels[:, 1]
        return ppg, sbp, dbp
    
    def resnet_block(input_tensor, filters, kernel_size=3, strides=1, use_batch_norm=True):
        x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer="he_normal")(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        if strides != 1 or input_tensor.shape[-1] != filters:
            input_tensor = Conv1D(filters, kernel_size=1, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
            if use_batch_norm:
                input_tensor = BatchNormalization()(input_tensor)
        
        x = Add()([x, input_tensor])
        x = Activation("relu")(x)
        return x
    
    def ResNet1D(input_shape, num_blocks, filters, num_outputs=2):
        X_input = Input(shape=input_shape)
        x = Conv1D(filters[0], kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(X_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        for i, f in enumerate(filters):
            for j in range(num_blocks[i]):
                strides = 2 if j == 0 and i > 0 else 1  
                x = resnet_block(x, f, strides=strides)
        x = GlobalAveragePooling1D()(x)
        SBP = Dense(1, activation="relu", name="SBP")(x)
        DBP = Dense(1, activation="relu", name="DBP")(x)
        model = Model(inputs=X_input, outputs=[SBP, DBP], name="ResNet1D")
        return model
    
    # Load and preprocess the data
    filename = "./Mimic.h5" 
    ppg, labels, subject_idx = load_dataset(filename, num_samples=10000)  
    ppg, sbp, dbp = preprocess_data(ppg, labels)
    
    # Split the data
    ppg_train, ppg_val, sbp_train, sbp_val, dbp_train, dbp_val = train_test_split(
        ppg, sbp, dbp, test_size=0.2, random_state=42
    )
    
    # Define and compile the model
    input_shape = (875, 1)  
    num_blocks = [2, 2, 2]  
    filters = [64, 128, 256] 
    model = ResNet1D(input_shape, num_blocks, filters, num_outputs=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"SBP": "mse", "DBP": "mse"},
        metrics={"SBP": "mae", "DBP": "mae"}
    )
    
    # Train the model
    history = model.fit(
        ppg_train,
        {"SBP": sbp_train, "DBP": dbp_train},
        validation_data=(ppg_val, {"SBP": sbp_val, "DBP": dbp_val}),
        epochs=15,
        batch_size=32
    )
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Training and saving model...")
        save_trained_model()
    else:
        print(f"Model already exists at {MODEL_PATH}")