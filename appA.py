# استيراد المكتبات
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # مكتبة لتحجيم البيانات

# تحميل مجموعة البيانات
data = load_breast_cancer()  # تحميل بيانات سرطان الثدي
x = data.data                # المتغيرات المستقلة
y = data.target              # المتغير التابع (الهدف)

# تعريف المتغيرات الأساسية
random_state = 20            # لتحديد عشوائية ثابتة لأغراض التكرار
test_size = 0.15              # حجم البيانات المخصصة للاختبار (20%)

# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

# تحجيم البيانات باستخدام StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # تحجيم بيانات التدريب
x_test_scaled = scaler.transform(x_test)        # تحجيم بيانات الاختبار

# تعريف قواميس النماذج
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=200),  # زيادة max_iter إلى 200
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=random_state)
}

# دالة لتدريب وتقييم كل نموذج
def train_and_evaluate_model(model_name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)                # تدريب النموذج باستخدام بيانات التدريب
    y_pred = model.predict(x_test)             # توقع النتائج باستخدام بيانات الاختبار
    accuracy = accuracy_score(y_test, y_pred)  # حساب دقة النموذج

    # حفظ دقة النموذج في ملف
    with open("metrics.txt", "a") as file:  # "a" للإلحاق بالملف
        file.write(f"Model: {model_name}\n")   # كتابة اسم النموذج في الملف
        file.write(f"Accuracy: {accuracy:.4f}\n")  # كتابة دقة النموذج بتنسيق أربع خانات عشرية
        file.write("-"*40 + "\n")              # فاصل لتحديد نهاية النموذج
    return accuracy                            # إرجاع دقة النموذج

# تمرير كل نموذج لتدريب وتقييم كل منها
for model_name, model in models.items():
    # استخدام البيانات المحجمة لكل نموذج
    accuracy = train_and_evaluate_model(model_name, model, x_train_scaled, x_test_scaled, y_train, y_test)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
