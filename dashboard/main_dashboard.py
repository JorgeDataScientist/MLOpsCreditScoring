import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
import matplotlib.image as mpimg
import requests
import os

# Importar el adaptador de preprocess
from preprocess_adapter import preprocess_single_record

# Configuración inicial
st.set_page_config(page_title="Intelligent Credit Scoring Pipeline", layout="wide")
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models/model_1/rf_model.pkl"
DATA_PATH = BASE_DIR / "data/raw/train.csv"
INFORME_PATH = BASE_DIR / "informe/model_1/informe.html"

# Descargar archivos grandes desde DagsHub
def download_file(url, dest_path):
    if not dest_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)

# Descarga archivos desde DagsHub
download_file("https://dagshub.com/JorgeDataScientist/MLOps_CreditScore/raw/master/models/model_1/rf_model.pkl", MODEL_PATH)
download_file("https://dagshub.com/JorgeDataScientist/MLOps_CreditScore/raw/master/data/raw/train.csv", DATA_PATH)
download_file("https://dagshub.com/JorgeDataScientist/MLOps_CreditScore/src/master/informe/model_1/informe.html", INFORME_PATH)

# Título y descripción
st.title("Intelligent Credit Scoring Pipeline Dashboard 🎉")
st.markdown("¡Explora los resultados del modelo de scoring crediticio de Jorge! 🚀 Creado por Jorge, 2025 😄")

# Barra lateral
with st.sidebar:
    st.header("Credit Scoring Pipeline")
    st.markdown("**Creado por Jorge Luis Garcia**")
    if st.button("LinkedIn", use_container_width=True):
        st.markdown('<meta http-equiv="refresh" content="0;url=https://www.linkedin.com/in/jorgedatascientistmlops/" target="_blank">', unsafe_allow_html=True)
    st.markdown("[Dagshub](https://dagshub.com/JorgeDataScientist)", unsafe_allow_html=True)
    st.markdown("[Sitio Web](https://jorgedatascientist.github.io/portafolio/)", unsafe_allow_html=True)
    st.markdown("[Docker Hub](https://hub.docker.com/u/jorgedatascientist)", unsafe_allow_html=True)
    st.markdown("**Email**: jorgeluisdatascientist@gmail.com")

# Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Métricas", "📈 Gráficos", "🗃️ Dataset", "🔍 Informe", "🧮 Predicciones"])

# Pestaña 1: Métricas
with tab1:
    st.subheader("Métricas del Modelo (Model_1)")
    metrics_path = BASE_DIR / "metrics/model_1/metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.write("Métricas del modelo:")
        st.dataframe(metrics)
        numeric_cols = metrics.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            metrics_numeric = metrics[numeric_cols].iloc[0]
            fig = px.bar(
                x=numeric_cols,
                y=metrics_numeric,
                title="Comparación de Métricas",
                labels={"x": "Métrica", "y": "Valor"},
            )
            st.plotly_chart(fig)
        else:
            st.warning("No se encontraron columnas numéricas en metrics.csv para graficar.")
        st.markdown("Estas métricas muestran el rendimiento del RandomForestClassifier entrenado en train.csv.")
    else:
        st.error("No se encontró metrics.csv en metrics/model_1/")

# Pestaña 2: Gráficos
with tab2:
    st.subheader("Gráficos de Rendimiento (Model_1)")
    graphics_path = BASE_DIR / "graphics/model_1"
    for img_name in ["confusion_matrix.png", "metrics_bar.png"]:
        img_path = graphics_path / img_name
        if img_path.exists():
            img = mpimg.imread(img_path)
            st.image(img, caption=img_name.replace(".png", "").replace("_", " ").title(), width=675)
        else:
            st.error(f"No se encontró {img_name} en graphics/model_1/")

# Pestaña 3: Dataset
with tab3:
    st.subheader("Exploración de train.csv")
    data_path = DATA_PATH
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.write("Dataset completo:")
        st.dataframe(df)
    else:
        st.error("No se encontró train.csv en data/raw/")

# Pestaña 4: Informe
with tab4:
    st.subheader("Reporte de Análisis")
    informe_path = INFORME_PATH
    if informe_path.exists():
        with open(informe_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.error("No se encontró informe.html en informe/model_1/")

# Pestaña 5: Predicciones
with tab5:
    st.subheader("Hacer Predicciones con Model_1")
    model_path = MODEL_PATH
    if not model_path.exists():
        st.error("No se encontró rf_model.pkl en models/model_1/")
        st.stop()

    st.markdown("Ingresa los datos del cliente:")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1, help="Edad del cliente")
            monthly_salary = st.number_input("Salario Mensual en Mano", min_value=0.0, value=5000.0, step=100.0, help="Salario neto mensual")
            num_credit_cards = st.number_input("Número de Tarjetas de Crédito", min_value=0, value=3, step=1)
            interest_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, value=5.0, step=0.1)
            outstanding_debt = st.number_input("Deuda Pendiente", min_value=0.0, value=2000.0, step=100.0)
            num_bank_accounts = st.number_input("Número de Cuentas Bancarias", min_value=0, value=2, step=1)
            num_of_loan = st.number_input("Número de Préstamos", min_value=0, value=1, step=1)
        with col2:
            delay_from_due_date = st.number_input("Días de Retraso desde la Fecha de Vencimiento", min_value=0, value=0, step=1)
            num_delayed_payments = st.number_input("Número de Pagos Retrasados", min_value=0, value=0, step=1)
            credit_history_age = st.number_input("Edad del Historial Crediticio (en meses)", min_value=0.0, value=24.0, step=1.0)
            total_emi_per_month = st.number_input("Total de Cuotas Mensuales", min_value=0.0, value=500.0, step=10.0)
            monthly_balance = st.number_input("Saldo Mensual", min_value=0.0, value=3000.0, step=100.0)
            amount_invested_monthly = st.number_input("Cantidad Invertida Mensualmente", min_value=0.0, value=100.0, step=10.0)

        occupation = st.selectbox(
            "Ocupación",
            ["Engineer", "Developer", "Doctor", "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer", "Architect"],
            index=0,
        )
        payment_behaviour = st.selectbox(
            "Comportamiento de Pago",
            [
                "High_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Small_value_payments",
            ],
            index=0,
        )
        credit_mix = st.selectbox("Mezcla Crediticia", ["Good", "Standard", "Bad"], index=0)
        payment_of_min_amount = st.selectbox("Pago del Monto Mínimo", ["Yes", "No"], index=0)

        submitted = st.form_submit_button("Predecir")
        if submitted:
            input_data = pd.DataFrame(
                {
                    "Age": [age],
                    "Monthly_Inhand_Salary": [monthly_salary],
                    "Num_Credit_Card": [num_credit_cards],
                    "Interest_Rate": [interest_rate],
                    "Outstanding_Debt": [outstanding_debt],
                    "Delay_from_due_date": [delay_from_due_date],
                    "Num_of_Delayed_Payment": [num_delayed_payments],
                    "Credit_History_Age": [credit_history_age],
                    "Total_EMI_per_month": [total_emi_per_month],
                    "Monthly_Balance": [monthly_balance],
                    "Amount_invested_monthly": [amount_invested_monthly],
                    "Occupation": [occupation],
                    "Payment_Behaviour": [payment_behaviour],
                    "Credit_Mix": [credit_mix],
                    "Payment_of_Min_Amount": [payment_of_min_amount],
                    "Num_Bank_Accounts": [num_bank_accounts],
                    "Num_of_Loan": [num_of_loan],
                }
            )

            try:
                processed_data = preprocess_single_record(input_data)
                model = joblib.load(model_path)
                prediction = model.predict(processed_data)[0]
                if prediction == "Good":
                    st.success(f"Predicción: **Good** 🎉")
                elif prediction == "Poor":
                    st.error(f"Predicción: **Poor** 😟")
                else:
                    st.warning(f"Predicción: **Standard** ⚖️")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")