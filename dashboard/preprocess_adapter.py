import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Agregar src al path para importar funciones de preprocess.py
sys.path.append(str(Path(__file__).parent.parent / "src"))
from preprocess import clean_data, transform_data, create_new_features, rename_columns, strip_strings, handle_missing_values, filter_minimum_values, filter_by_age, filter_by_age_credit_ratio, drop_columns, apply_encoding_rules, select_final_columns

# Clase auxiliar para simular objetos de Hydra
class SimpleConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Configuración estática (simula config/process/preprocess.yaml)
CONFIG = {
    "process": {
        "translations": {
            "Age": "Edad",
            "Monthly_Inhand_Salary": "Salario_Mensual",
            "Num_Credit_Card": "Num_Tarjetas_Credito",
            "Interest_Rate": "Tasa_Interes",
            "Delay_from_due_date": "Retraso_Pago",
            "Num_of_Delayed_Payment": "Num_Pagos_Retrasados",
            "Changed_Credit_Limit": "Cambio_Limite_Credito",
            "Num_Credit_Inquiries": "Num_Consultas_Credito",
            "Outstanding_Debt": "Deuda_Pendiente",
            "Credit_History_Age": "Edad_Historial_Credito",
            "Total_EMI_per_month": "Total_Cuota_Mensual",
            "Amount_invested_monthly": "Inversion_Mensual",
            "Monthly_Balance": "Saldo_Mensual",
            "Payment_Behaviour": "Comportamiento_de_Pago",
            "Credit_Mix": "Mezcla_Crediticia",
            "Payment_of_Min_Amount": "Pago_Minimo",
            "Occupation": "Ocupacion",
            "Credit_Score": "Puntaje_Credito",
            "Num_Bank_Accounts": "Num_Cuentas_Bancarias",
            "Num_of_Loan": "Num_Prestamos"
        },
        "cleaning": {
            "min_age": 18,
            "max_age_credit_ratio": 100,
            "drop_columns": [
                "ID",
                "Customer_ID",
                "Month",
                "Name",
                "SSN",
                "Type_of_Loan",
                "Credit_Utilization_Ratio",
                "Annual_Income",
                "Num_Bank_Accounts",
                "Num_of_Loan"
            ]
        },
        "encoding": {
            "Comportamiento_de_Pago": {"drop": None},
            "Mezcla_Crediticia": {"drop": None},
            "Pago_Minimo": {"drop": None},
            "Ocupacion": {"drop": None}
        },
        "new_features": [
            SimpleConfig(name="debt_to_income", formula={}),
            SimpleConfig(name="payment_to_income", formula={}),
            SimpleConfig(name="credit_history_ratio", formula={})
        ],
        "target": "Puntaje_Credito",
        "test_size": 0.2,
        "random_state": 42
    }
}

def preprocess_single_record(input_data: pd.DataFrame) -> pd.DataFrame:
    """Procesa un solo registro para predicción con rf_model.pkl.

    Args:
        input_data: DataFrame con columnas de train.csv.

    Returns:
        DataFrame procesado listo para el modelo.
    """
    # Crear una copia del input
    df = input_data.copy()

    # Aplicar limpiezas de preprocess.py
    df = rename_columns(df, CONFIG["process"]["translations"])
    df = strip_strings(df)
    df = handle_missing_values(df)
    df = filter_minimum_values(df)
    df = filter_by_age(df, CONFIG["process"]["cleaning"]["min_age"])
    df = filter_by_age_credit_ratio(df, CONFIG["process"]["cleaning"]["max_age_credit_ratio"])
    df = drop_columns(df, CONFIG["process"]["cleaning"]["drop_columns"])

    # Añadir columnas faltantes con valores por defecto
    if "Cambio_Limite_Credito" not in df.columns:
        df["Cambio_Limite_Credito"] = 0.0
    if "Num_Consultas_Credito" not in df.columns:
        df["Num_Consultas_Credito"] = 0

    # Aplicar transformaciones (codificación)
    df = apply_encoding_rules(df, CONFIG["process"]["encoding"])

    # Crear nuevas features
    df = create_new_features(df, CONFIG["process"]["new_features"])

    # Lista de columnas esperadas por el modelo (del JSON)
    expected_columns = [
        "Edad",
        "Salario_Mensual",
        "Num_Tarjetas_Credito",
        "Tasa_Interes",
        "Retraso_Pago",
        "Num_Pagos_Retrasados",
        "Cambio_Limite_Credito",
        "Num_Consultas_Credito",
        "Deuda_Pendiente",
        "Edad_Historial_Credito",
        "Total_Cuota_Mensual",
        "Inversion_Mensual",
        "Saldo_Mensual",
        "Comportamiento_de_Pago_High_spent_Large_value_payments",
        "Comportamiento_de_Pago_High_spent_Medium_value_payments",
        "Comportamiento_de_Pago_High_spent_Small_value_payments",
        "Comportamiento_de_Pago_Low_spent_Large_value_payments",
        "Comportamiento_de_Pago_Low_spent_Medium_value_payments",
        "Comportamiento_de_Pago_Low_spent_Small_value_payments",
        "Mezcla_Crediticia_Bad",
        "Mezcla_Crediticia_Good",
        "Mezcla_Crediticia_Standard",
        "Pago_Minimo_No",
        "Pago_Minimo_Yes",
        "Ocupacion_Architect",
        "Ocupacion_Developer",
        "Ocupacion_Doctor",
        "Ocupacion_Engineer",
        "Ocupacion_Entrepreneur",
        "Ocupacion_Journalist",
        "Ocupacion_Lawyer",
        "Ocupacion_Manager",
        "Ocupacion_Mechanic",
        "Ocupacion_Media_Manager",
        "Ocupacion_Musician",
        "Ocupacion_Scientist",
        "Ocupacion_Teacher",
        "Ocupacion_Writer",
        "debt_to_income",
        "payment_to_income",
        "credit_history_ratio"
    ]

    # Asegurar que todas las columnas esperadas estén presentes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Seleccionar solo las columnas esperadas en el orden correcto
    df = df[expected_columns]

    return df