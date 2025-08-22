# Deep Learning Example Project

โปรเจคตัวอย่างการใช้งาน Deep Learning พร้อมสร้างข้อมูลตัวอย่าง CSV

## 📋 รายละเอียด

โปรเจคนี้ประกอบด้วยตัวอย่าง 2 ปัญหา:

1. **Regression Problem**: ทำนายราคาบ้านจาก features ต่างๆ
2. **Classification Problem**: ทำนายการเกิดฝนจากข้อมูลสภาพอากาศ

## 🚀 การเริ่มต้น

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. รันตัวอย่าง:
```bash
python run_example.py
```
หรือ
```bash
python deep_learning_example.py
```

## 📊 ไฟล์ที่จะถูกสร้าง

เมื่อรันโปรแกรมเสร็จจะได้ไฟล์:

### ข้อมูล (CSV)
- `house_price_dataset.csv` - ข้อมูลราคาบ้าน
- `weather_classification_dataset.csv` - ข้อมูลสภาพอากาศ

### โมเดล
- `house_price_model.h5` - โมเดลทำนายราคาบ้าน
- `weather_prediction_model.h5` - โมเดลทำนายฝน

### กราฟ
- `regression_results.png` - ผลลัพธ์ regression
- `classification_results.png` - ผลลัพธ์ classification

## 🎯 Features

### Regression Model (ทำนายราคาบ้าน)
- Input features: อายุ, รายได้, ปีการศึกษา, ประสบการณ์
- Output: ราคาบ้าน
- Architecture: 4-layer neural network
- Metrics: MSE, R²

### Classification Model (ทำนายฝน)
- Input features: อุณหภูมิ, ความชื้น, ความเร็วลม, ความกดอากาศ
- Output: จะเกิดฝนหรือไม่ (0/1)
- Architecture: 4-layer neural network with sigmoid
- Metrics: Accuracy, Classification Report

## 🛠️ เทคโนโลยีที่ใช้

- **TensorFlow/Keras**: Deep Learning Framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization

## 📁 โครงสร้างโปรเจค

```
basic-deep/
├── main.py                    # Entry point
├── run_example.py             # Example runner
├── deep_learning_example.py   # Main example code
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── pyproject.toml            # Project configuration
```

## 🔍 รายละเอียดโค้ด

### การสร้างข้อมูล
- สุ่มข้อมูลด้วย NumPy
- สร้างความสัมพันธ์เชิงเส้นและไม่เชิงเส้น
- บันทึกเป็นไฟล์ CSV

### การฝึกโมเดล
- แบ่งข้อมูล train/test (80/20)
- Standardization ข้อมูล
- Early stopping และ Dropout
- Validation split สำหรับ monitoring

### การประเมินผล
- Regression: MSE, R², Scatter plot
- Classification: Accuracy, Classification report
- Training curves visualization

## 🎨 การปรับแต่ง

คุณสามารถปรับแต่งได้:
- จำนวนข้อมูล (`n_samples`)
- Architecture ของโมเดล
- Hyperparameters (learning rate, epochs, etc.)
- การสร้างข้อมูล (features, relationships)

## 📞 การใช้งาน

รันโปรแกรมแล้วดูผลลัพธ์ใน console และไฟล์กราฟที่สร้างขึ้น!
