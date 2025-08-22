import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

def create_regression_dataset():
    """สร้างข้อมูลตัวอย่างสำหรับ regression problem"""
    np.random.seed(42)
    n_samples = 1000
    
    # สร้างข้อมูล features
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
    }
    
    # สร้าง target variable (house price) ตาม features
    house_price = (
        data['income'] * 3 +
        data['age'] * 500 +
        data['education_years'] * 2000 +
        data['experience'] * 1000 +
        np.random.normal(0, 10000, n_samples)
    )
    
    data['house_price'] = np.maximum(house_price, 50000)  # ราคาบ้านไม่น้อยกว่า 50,000
    
    df = pd.DataFrame(data)
    df.to_csv('house_price_dataset.csv', index=False)
    return df

def create_classification_dataset():
    """สร้างข้อมูลตัวอย่างสำหรับ classification problem"""
    np.random.seed(42)
    n_samples = 1000
    
    # สร้างข้อมูล features
    data = {
        'temperature': np.random.normal(25, 10, n_samples),
        'humidity': np.random.normal(60, 20, n_samples),
        'wind_speed': np.random.normal(10, 5, n_samples),
        'pressure': np.random.normal(1013, 50, n_samples),
    }
    
    # สร้าง target variable (rain prediction) ตาม features
    rain_probability = (
        (data['humidity'] > 70) * 0.3 +
        (data['temperature'] < 20) * 0.2 +
        (data['wind_speed'] > 15) * 0.2 +
        (data['pressure'] < 1000) * 0.3
    )
    
    data['will_rain'] = np.random.binomial(1, rain_probability, n_samples)
    
    df = pd.DataFrame(data)
    df.to_csv('weather_classification_dataset.csv', index=False)
    return df

def build_regression_model(input_shape):
    """สร้าง neural network สำหรับ regression"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

def build_classification_model(input_shape):
    """สร้าง neural network สำหรับ classification"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_regression_model():
    """ฝึก regression model"""
    print("=== Regression Example: House Price Prediction ===")
    
    # สร้างและโหลดข้อมูล
    df = create_regression_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(f"\nDataset statistics:\n{df.describe()}")
    
    # แยกข้อมูล features และ target
    X = df[['age', 'income', 'education_years', 'experience']]
    y = df['house_price']
    
    # แบ่งข้อมูล train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale ข้อมูล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # สร้างและฝึก model
    model = build_regression_model(X_train_scaled.shape[1])
    print(f"\nModel architecture:")
    model.summary()
    
    # ฝึก model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # ประเมินผล
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== Regression Results ===")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, scaler

def train_classification_model():
    """ฝึก classification model"""
    print("\n=== Classification Example: Weather Rain Prediction ===")
    
    # สร้างและโหลดข้อมูล
    df = create_classification_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(f"\nClass distribution:\n{df['will_rain'].value_counts()}")
    
    # แยกข้อมูล features และ target
    X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
    y = df['will_rain']
    
    # แบ่งข้อมูล train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale ข้อมูล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # สร้างและฝึก model
    model = build_classification_model(X_train_scaled.shape[1])
    print(f"\nModel architecture:")
    model.summary()
    
    # ฝึก model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # ประเมินผล
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Classification Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, scaler

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 Deep Learning Examples with Sample Data")
    print("=" * 50)
    
    # ตั้งค่า TensorFlow
    tf.random.set_seed(42)
    
    try:
        # 1. Regression Example
        reg_model, reg_scaler = train_regression_model()
        
        # 2. Classification Example  
        clf_model, clf_scaler = train_classification_model()
        
        # บันทึก models
        reg_model.save('house_price_model.h5')
        clf_model.save('weather_prediction_model.h5')
        
        print("\n✅ Training completed successfully!")
        print("📁 Files created:")
        print("   - house_price_dataset.csv")
        print("   - weather_classification_dataset.csv")
        print("   - house_price_model.h5")
        print("   - weather_prediction_model.h5")
        print("   - regression_results.png")
        print("   - classification_results.png")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
