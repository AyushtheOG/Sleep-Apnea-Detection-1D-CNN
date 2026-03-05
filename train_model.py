import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_1d_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading dataset...")
    df = pd.read_pickle('/content/Dataset/breathing_dataset.pkl')
        
    X = np.stack(df['Signal_Matrix'].values)
    y = df['Label'].values
    groups = df['Participant'].values
    
    logo = LeaveOneGroupOut()
    accuracies, precisions, recalls = [], [], []
    all_y_true, all_y_pred = [], []
    
    print("\nStarting Leave-One-Participant-Out Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = groups[test_idx][0]
        print(f"\n--- Fold {fold + 1}: Testing on {test_participant} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = build_1d_cnn(input_shape=(X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f}")

    print("\n" + "="*40)
    print("FINAL LOOCV RESULTS (Average across all participants):")
    print(f"Mean Accuracy:  {np.mean(accuracies):.2f}")
    print(f"Mean Precision: {np.mean(precisions):.2f}")
    print(f"Mean Recall:    {np.mean(recalls):.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
