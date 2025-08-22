# 📋 ขั้นตอนการสร้างโปรเจค Deep Learning

เอกสารนี้อธิบายขั้นตอนการสร้างโปรเจค Deep Learning พร้อมข้อมูลตัวอย่าง CSV ตั้งแต่เริ่มต้นจนสำเร็จ

## 🎯 เป้าหมาย
- สร้างโปรเจค Deep Learning ที่สามารถทำ Regression และ Classification
- สร้างข้อมูลตัวอย่าง CSV อัตโนมัติ
- ฝึกโมเดล Neural Network และประเมินผล
- บันทึกผลลัพธ์และ visualizations
- อัปโหลดโปรเจคขึ้น GitHub

---

## 📋 สารบัญ

1. [การตั้งค่าโปรเจคเริ่มต้น](#1-การตั้งค่าโปรเจคเริ่มต้น)
2. [การติดตั้ง Dependencies](#2-การติดตั้ง-dependencies)
3. [การสร้างโค้ด Deep Learning](#3-การสร้างโค้ด-deep-learning)
4. [การสร้างไฟล์เสริม](#4-การสร้างไฟล์เสริม)
5. [การทดสอบโปรแกรม](#5-การทดสอบโปรแกรม)
6. [การอัปโหลดขึ้น GitHub](#6-การอัปโหลดขึ้น-github)
7. [การใช้งานโปรเจค](#7-การใช้งานโปรเจค)

---

## 1. การตั้งค่าโปรเจคเริ่มต้น

### 1.1 สร้างโฟลเดอร์โปรเจค
```bash
mkdir basic-deep
cd basic-deep
```

### 1.2 เริ่มต้น Git Repository
```bash
git init
git branch -M main
```

### 1.3 สร้างไฟล์ .gitignore
```bash
echo "*.pyc
__pycache__/
.env
.venv/
.DS_Store
*.log" > .gitignore
```

### 1.4 สร้างไฟล์ pyproject.toml พื้นฐาน
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "basic-deep"
version = "0.1.0"
description = "Basic Deep Learning Examples"
```

---

## 2. การติดตั้ง Dependencies

### 2.1 ตั้งค่า Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# หรือ .venv\Scripts\activate  # Windows
```

### 2.2 สร้างไฟล์ requirements.txt
```txt
tensorflow>=2.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 2.3 ติดตั้ง packages
```bash
pip install -r requirements.txt
```

---

## 3. การสร้างโค้ด Deep Learning

### 3.1 สร้างไฟล์ main.py (จุดเริ่มต้น)
```python
"""
Deep Learning Project Main Entry Point
จุดเริ่มต้นของโปรเจค Deep Learning
"""

def main():
    print("🚀 Welcome to Deep Learning Project!")
    print("=" * 40)
    print("Available examples:")
    print("1. Run: python run_example.py")
    print("2. Or: python deep_learning_example.py")
    print("-" * 40)
    print("This will:")
    print("✓ Create sample CSV datasets")
    print("✓ Train regression model (house price prediction)")
    print("✓ Train classification model (weather prediction)")  
    print("✓ Generate visualizations and save models")
    print("=" * 40)

if __name__ == "__main__":
    main()
```

### 3.2 สร้างไฟล์ deep_learning_example.py (โค้ดหลัก)

**ส่วนที่ 1: Import libraries และฟังก์ชันสร้างข้อมูล**
```python
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
    
    data['house_price'] = np.maximum(house_price, 50000)
    df = pd.DataFrame(data)
    df.to_csv('house_price_dataset.csv', index=False)
    return df
```

**ส่วนที่ 2: ฟังก์ชันสร้างข้อมูล Classification**
```python
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
    
    # สร้าง target variable (rain prediction)
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
```

**ส่วนที่ 3: ฟังก์ชันสร้างโมเดล**
```python
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
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

**ส่วนที่ 4: ฟังก์ชันฝึกโมเดล**
```python
def train_regression_model():
    """ฝึก regression model"""
    print("=== Regression Example: House Price Prediction ===")
    
    # สร้างและโหลดข้อมูล
    df = create_regression_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
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
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
    
    # ประเมินผล
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # บันทึกกราฟ
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
```

**ส่วนที่ 5: ฟังก์ชัน main**
```python
def main():
    """ฟังก์ชันหลัก"""
    print("🚀 Deep Learning Examples with Sample Data")
    print("=" * 50)
    
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
```

### 3.3 สร้างไฟล์ run_example.py (ไฟล์รันง่าย)
```python
"""
Simple Deep Learning Example Runner
เรียกใช้งาน deep learning example
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_learning_example import main

def run_example():
    """เรียกใช้งานตัวอย่าง deep learning"""
    print("🎯 Starting Deep Learning Examples...")
    print("This example will create:")
    print("  1. Sample CSV datasets")
    print("  2. Train neural networks for regression and classification")
    print("  3. Save trained models and visualizations")
    print("-" * 60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required packages are installed")

if __name__ == "__main__":
    run_example()
```

---

## 4. การสร้างไฟล์เสริม

### 4.1 สร้างไฟล์ README.md
```markdown
# Deep Learning Example Project

โปรเจคตัวอย่างการใช้งาน Deep Learning พร้อมสร้างข้อมูลตัวอย่าง CSV

## 🚀 การเริ่มต้น

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. รันตัวอย่าง:
```bash
python run_example.py
```

## 📊 ไฟล์ที่จะถูกสร้าง

- `house_price_dataset.csv` - ข้อมูลราคาบ้าน
- `weather_classification_dataset.csv` - ข้อมูลสภาพอากาศ
- `house_price_model.h5` - โมเดลทำนายราคาบ้าน
- `weather_prediction_model.h5` - โมเดลทำนายฝน
- `regression_results.png` - ผลลัพธ์ regression
- `classification_results.png` - ผลลัพธ์ classification
```

### 4.2 อัปเดต pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "basic-deep"
version = "0.1.0"
description = "Basic Deep Learning Examples with Sample CSV Data"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "tensorflow>=2.13.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0"
]

[project.urls]
Homepage = "https://github.com/NPKMUTNB/basic-deep"
Repository = "https://github.com/NPKMUTNB/basic-deep.git"
```

---

## 5. การทดสอบโปรแกรม

### 5.1 ตรวจสอบไฟล์
```bash
ls -la
# ควรเห็นไฟล์: main.py, deep_learning_example.py, run_example.py, requirements.txt, README.md
```

### 5.2 รันโปรแกรม
```bash
python main.py
# ควรแสดงข้อความต้อนรับ

python run_example.py
# ควรรันได้สำเร็จและสร้างไฟล์ผลลัพธ์
```

### 5.3 ตรวจสอบไฟล์ผลลัพธ์
```bash
ls -la *.csv *.h5 *.png
# ควรเห็น: house_price_dataset.csv, weather_classification_dataset.csv, 
#         house_price_model.h5, weather_prediction_model.h5,
#         regression_results.png, classification_results.png
```

---

## 6. การอัปโหลดขึ้น GitHub

### 6.1 เตรียมไฟล์สำหรับ Git
```bash
git add .
git status
# ตรวจสอบไฟล์ที่จะ commit
```

### 6.2 Commit การเปลี่ยนแปลง
```bash
git commit -m "Initial commit: Add deep learning examples with sample CSV data generation"
```

### 6.3 สร้าง GitHub Repository
```bash
# ใช้ GitHub CLI หรือสร้างผ่านเว็บ GitHub
gh repo create basic-deep --public --description "Basic Deep Learning Examples with Sample CSV Data"
```

### 6.4 เชื่อมต่อกับ GitHub
```bash
git remote add origin https://github.com/USERNAME/basic-deep.git
git push -u origin main
```

### 6.5 Push ไฟล์ผลลัพธ์
```bash
git add *.csv *.h5 *.png
git commit -m "Add generated datasets, trained models, and visualizations"
git push
```

---

## 7. การใช้งานโปรเจค

### 7.1 Clone Repository
```bash
git clone https://github.com/USERNAME/basic-deep.git
cd basic-deep
```

### 7.2 ตั้งค่า Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 7.3 รันตัวอย่าง
```bash
python run_example.py
```

### 7.4 ดูผลลัพธ์
- ไฟล์ CSV: เปิดด้วย Excel หรือ text editor
- ไฟล์ PNG: เปิดดูกราฟผลลัพธ์
- ไฟล์ H5: โมเดลที่สามารถโหลดกลับมาใช้ได้

---

## 🔧 การปรับแต่ง

### เปลี่ยนจำนวนข้อมูล
แก้ไขค่า `n_samples` ในฟังก์ชัน `create_regression_dataset()` และ `create_classification_dataset()`

### เปลี่ยน Architecture โมเดล
แก้ไขฟังก์ชัน `build_regression_model()` และ `build_classification_model()`

### เปลี่ยน Hyperparameters
แก้ไขค่า `epochs`, `batch_size` ในฟังก์ชัน `model.fit()`

---

## 🎯 สรุป

โปรเจคนี้แสดงให้เห็นถึงขั้นตอนการสร้าง Deep Learning application ที่สมบูรณ์:

1. **การจัดการข้อมูล**: สร้างข้อมูลตัวอย่างและบันทึกเป็น CSV
2. **การสร้างโมเดล**: ใช้ TensorFlow/Keras สร้าง Neural Networks
3. **การฝึกโมเดล**: ฝึกโมเดลทั้ง Regression และ Classification
4. **การประเมินผล**: วัดประสิทธิภาพและสร้าง visualizations
5. **การจัดเก็บ**: บันทึกโมเดลและผลลัพธ์
6. **การแชร์**: อัปโหลดขึ้น GitHub เพื่อแชร์กับคนอื่น

สามารถนำไปต่อยอดพัฒนาเป็นโปรเจคที่ซับซ้อนมากขึ้นได้ตามต้องการ!

---

## 📞 ติดต่อ

หากมีคำถามหรือข้อเสนอแนะ สามารถติดต่อได้ผ่าน:
- GitHub Issues: [https://github.com/USERNAME/basic-deep/issues](https://github.com/USERNAME/basic-deep/issues)
- Email: your.email@example.com

---

*เอกสารนี้สร้างขึ้นเพื่อการศึกษาและเป็นแนวทางในการสร้างโปรเจค Deep Learning*
