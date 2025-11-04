import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import json, os

# ✅ UI Components
from components.ui_theme import apply_theme, card, toast_success, toast_warning
from components.charts import (
    class_probability_bar, feature_importance_bar, confidence_gauge,
    radar_metrics, confusion_matrix_animated, hist_with_box, box_by_class,
    correlation_heatmap, class_distribution_donut
)
from components.layout import header, sticky_sidebar_tips, footer

# ------------------------------------------------------------
# ✅ Streamlit Page + Theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven Water Stress Management in Plants",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()

# ------------------------------------------------------------
# ✅ Load Models & Artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")

        model_files = {
            "Hybrid (Voting)": "models/hybrid_(voting)_model.pkl",
            "Gradient Boosting": "models/gradient_boosting_model.pkl",
            "Random Forest": "models/random_forest_model.pkl",
            "SVM": "models/svm_model.pkl",
            "Decision Tree": "models/decision_tree_model.pkl",
            "KNN": "models/knn_model.pkl",
            "Logistic Regression": "models/logistic_regression_model.pkl"
        }

        models = {name: joblib.load(path)
                  for name, path in model_files.items() if os.path.exists(path)}

        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)

        return models, scaler, le, metrics

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

models, scaler, le, metrics = load_artifacts()
if not all([models, scaler, le, metrics]):
    st.stop()

# ------------------------------------------------------------
# ✅ Feature Names
# ------------------------------------------------------------
feature_names = [
    'Soil_Moisture','Ambient_Temperature','Soil_Temperature','Humidity',
    'Light_Intensity','Soil_pH','Nitrogen_Level','Phosphorus_Level',
    'Potassium_Level','Chlorophyll_Content','Electrochemical_Signal'
]

# ------------------------------------------------------------
# ✅ Header & Sidebar Tips
# ------------------------------------------------------------
header("AI-Driven Water Stress Management in Plants", "Smart plant stress prediction dashboard")
sticky_sidebar_tips()

# ------------------------------------------------------------
# ✅ Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([" Prediction", " Model Comparison", " Data Visualization", " About"])

# ------------------------------------------------------------
# ✅ TAB 1 — Prediction
# ------------------------------------------------------------
with tab1:

    col1, col2 = st.columns((2,1))

    # Inputs
    with col1:
        with card("<span class='sensor-title'> Sensor Inputs</span>"):

            model_name = st.selectbox("Model", list(models.keys()))
            ranges = {
                'Soil_Moisture': (0,50,25),'Ambient_Temperature':(0,40,25),
                'Soil_Temperature':(0,40,20),'Humidity':(0,100,50),
                'Light_Intensity':(0,1000,500),'Soil_pH':(0,14,7),
                'Nitrogen_Level':(0,50,25),'Phosphorus_Level':(0,50,25),
                'Potassium_Level':(0,50,25),'Chlorophyll_Content':(0,50,30),
                'Electrochemical_Signal':(0,2,1)
            }
            cols = st.columns(3)
            user_input = {}
            for i, feat in enumerate(feature_names):
                mn,mx,df = ranges.get(feat,(0,100,50))
                user_input[feat] = cols[i % 3].slider(
                        label=feat,
                        min_value=float(mn),
                        max_value=float(mx),
                        value=float(df),
                        step=0.1
                    )

                

            predict = st.button("🔍 Predict", type="primary")

    # Results
    with col2:
        with card("<span class='sensor-title'> Prediction Results</span>"):

            if predict:
                X = pd.DataFrame([user_input])[feature_names]
                Xs = scaler.transform(X)

                model = models[model_name]
                p = model.predict(Xs)[0]
                proba = model.predict_proba(Xs)[0]
                label = le.inverse_transform([p])[0]
                conf = max(proba)*100

                # Display Result
                if label == "High Stress":
                    st.error(f"⚠️ {label}")
                    st.markdown("<div class='high-stress-pulse'></div>", unsafe_allow_html=True)
                    toast_warning("High Stress detected — check irrigation!")
                elif label == "Moderate Stress":
                    st.warning(f"⚡ {label}")
                else:
                    st.success(f"✅ {label}")
                    toast_success("Healthy Plant! 🌿")
                    try: st.balloons()
                    except: pass

                # Confidence meter
                st.plotly_chart(confidence_gauge(conf), use_container_width=True)

                # Probabilities
                st.markdown("#### Class Probabilities")
                st.plotly_chart(class_probability_bar(le.classes_, proba), use_container_width=True)

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    st.markdown("#### Key Features")
                    st.plotly_chart(
                        feature_importance_bar(model.feature_importances_, feature_names),
                        use_container_width=True
                    )

# ------------------------------------------------------------
# ✅ TAB 2 — Model Comparison
# ------------------------------------------------------------
with tab2:
    st.subheader("Performance Overview")

    model_pick = st.selectbox("Compare Model", list(metrics.keys()))
    data = metrics[model_pick]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(radar_metrics(data), use_container_width=True)

    with col2:
        cm = np.array(data["confusion_matrix"])
        st.plotly_chart(confusion_matrix_animated(cm, list(le.classes_)), use_container_width=True)

# ------------------------------------------------------------
# ✅ TAB 3 — Data Visualization
# ------------------------------------------------------------
with tab3:
    st.subheader("Dataset Insights")

    df = pd.read_csv("plant_health_data.csv")

    st.markdown("#### Class Distribution")
    st.plotly_chart(class_distribution_donut(df["Plant_Health_Status"]), use_container_width=True)

    feat = st.selectbox("Feature", feature_names)

    colA, colB = st.columns(2)
    colA.plotly_chart(hist_with_box(df, feat, "Plant_Health_Status"), use_container_width=True)
    colB.plotly_chart(box_by_class(df, feat, "Plant_Health_Status"), use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    num = df[feature_names]
    st.plotly_chart(correlation_heatmap(num), use_container_width=True)

# ------------------------------------------------------------
# ✅ TAB 4 — About
# ------------------------------------------------------------
with tab4:
    st.subheader("About This Project 🌿")

    st.markdown("""
### 🌱 AI-Driven Water Stress Management in Plants

Modern agriculture faces a silent challenge — plants often show stress **only when it's too late**.  
This project aims to solve that problem by **predicting plant health early** using smart sensors and machine learning.

We collect real-time environmental and nutritional parameters like soil moisture, temperature, pH, nutrients, and light exposure.  
These signals are then processed through advanced ML models to identify whether a plant is:

-  Healthy  
-  Moderately Stressed  
-  Highly Stressed  

By doing this, farmers and researchers can take **timely action**, improve productivity, and support sustainable farming.

---

###  Team Members

| Name | Roll No |
|------|--------|
| **Kirtthan Duvvi** | 22BCE0061 |
| **Kushal Sharma** | 22BCE2561 |
| **Akshar Varma** | 22BCT0372 |

---

###  Project Guide

**Under the guidance of:**  
**Dr. Perepi Raja Rajeshwari**

Thank you for the mentorship and continuous support throughout this project.

---

###  Project Goal

To build an intelligent system that:

- Analyzes plant sensor data in real time  
- Predicts stress condition with high accuracy  
- Provides actionable insights for precision farming  
- Uses hybrid machine learning models for improved performance  
- Offers clean, interactive, user-friendly visualization

---

###  Future Enhancements

-  IoT-based automated irrigation
-  Mobile companion app
-  Deep-learning sensor fusion

---

Vellore Institute of Technology — 2025

""")

footer()



