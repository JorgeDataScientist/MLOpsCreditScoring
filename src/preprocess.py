"""Procesa datos crudos para el modelo de Credit Scoring.

Este módulo carga datos desde data/raw/, aplica limpiezas, codifica variables
categóricas, crea nuevas features y divide el dataset en conjuntos de
entrenamiento y prueba, guardándolos en data/processed/.

Dependencias:
    - pandas: Para manipulación de datos.
    - numpy: Para cálculos numéricos.
    - sklearn: Para codificación y división de datos.
    - hydra: Para gestionar configuraciones.
    - pathlib: Para manejo de rutas portables.
    - logging: Para registro de eventos.
"""

import pandas as pd
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import logging

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(raw_path: str) -> pd.DataFrame:
    """Carga datos crudos desde un archivo CSV.

    Args:
        raw_path: Ruta al archivo CSV crudo.

    Returns:
        DataFrame con los datos cargados.
    """
    abs_path = str(Path(get_original_cwd()) / raw_path)
    logger.info(f"Cargando datos desde {abs_path}")
    return pd.read_csv(abs_path)


def rename_columns(df: pd.DataFrame, translations: dict) -> pd.DataFrame:
    """Renombra columnas según un diccionario de traducciones.

    Args:
        df: DataFrame original.
        translations: Diccionario con nombres originales y nuevos.

    Returns:
        DataFrame con columnas renombradas.
    """
    logger.info("Renombrando columnas")
    return df.rename(columns=translations)


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina espacios en blanco de valores tipo string.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame con strings limpiados.
    """
    logger.info("Limpiando espacios en strings")
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Maneja valores nulos imputando con la mediana para numéricos y 'unknown' para categóricos.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame sin valores nulos.
    """
    logger.info("Manejando valores nulos")
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in ["int64", "float64"]:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna("unknown")
    return df_clean


def filter_minimum_values(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra filas con valores mínimos en columnas específicas.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame filtrado.
    """
    logger.info("Filtrando valores mínimos")
    df_filtered = df.copy()
    if "Num_Cuentas_Bancarias" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Num_Cuentas_Bancarias"] > 0]
    if "Num_Prestamos" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Num_Prestamos"] > 0]
    return df_filtered


def filter_by_age(df: pd.DataFrame, min_age: int) -> pd.DataFrame:
    """Filtra filas por edad mínima.

    Args:
        df: DataFrame original.
        min_age: Edad mínima para filtrar.

    Returns:
        DataFrame con edades >= min_age.
    """
    logger.info(f"Filtrando por edad mínima: {min_age}")
    return df[df["Edad"] >= min_age]


def filter_by_age_credit_ratio(df: pd.DataFrame, max_ratio: float) -> pd.DataFrame:
    """Filtra filas según la relación edad/historial crediticio.

    Args:
        df: DataFrame original.
        max_ratio: Máximo valor permitido para la relación.

    Returns:
        DataFrame filtrado sin la columna temporal.
    """
    logger.info(f"Filtrando por relación edad/historial crediticio: {max_ratio}")
    df_filtered = df.copy()
    df_filtered["age_credit_ratio"] = df_filtered["Edad"] / (
        df_filtered["Edad_Historial_Credito"] / 12
    )
    df_filtered = df_filtered[df_filtered["age_credit_ratio"] <= max_ratio]
    return df_filtered.drop(columns=["age_credit_ratio"])


def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Elimina columnas especificadas.

    Args:
        df: DataFrame original.
        columns_to_drop: Lista de columnas a eliminar.

    Returns:
        DataFrame sin las columnas especificadas.
    """
    logger.info(f"Eliminando columnas: {columns_to_drop}")
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def clean_data(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Aplica limpiezas al dataset en pasos modulares.

    Args:
        df: DataFrame crudo.
        config: Configuración con reglas de limpieza.

    Returns:
        DataFrame limpio.
    """
    logger.info("Iniciando limpieza de datos")
    df_clean = df.copy()
    df_clean = rename_columns(df_clean, config.process.translations)
    df_clean = strip_strings(df_clean)
    df_clean = handle_missing_values(df_clean)
    df_clean = filter_minimum_values(df_clean)
    df_clean = filter_by_age(df_clean, config.process.cleaning.min_age)
    df_clean = filter_by_age_credit_ratio(df_clean, config.process.cleaning.max_age_credit_ratio)
    df_clean = drop_columns(df_clean, config.process.cleaning.drop_columns)
    return df_clean


def encode_categorical(df: pd.DataFrame, column: str, drop: str | None) -> pd.DataFrame:
    """Codifica una columna categórica con OneHotEncoding.

    Args:
        df: DataFrame original.
        column: Nombre de la columna a codificar.
        drop: Categoría a eliminar (ej. 'first') o None.

    Returns:
        DataFrame con la columna codificada.
    """
    logger.info(f"Codificando columna categórica: {column}")
    df_encoded = df.copy()
    ohe = OneHotEncoder(sparse_output=False, drop=drop, handle_unknown="ignore")
    encoded_data = ohe.fit_transform(df_encoded[[column]])
    if drop is not None and ohe.drop_idx_ is not None:
        encoded_cols = [
            f"{column}_{cat}"
            for i, cat in enumerate(ohe.categories_[0])
            if i != ohe.drop_idx_[0]
        ]
    else:
        encoded_cols = [f"{column}_{cat}" for cat in ohe.categories_[0]]
    df_encoded = pd.concat(
        [
            df_encoded,
            pd.DataFrame(encoded_data, columns=encoded_cols, index=df_encoded.index),
        ],
        axis=1,
    )
    return df_encoded.drop(columns=[column])


def apply_encoding_rules(df: pd.DataFrame, encoding_rules: dict) -> pd.DataFrame:
    """Aplica codificación a múltiples columnas según reglas.

    Args:
        df: DataFrame original.
        encoding_rules: Diccionario con columnas y parámetros de codificación.

    Returns:
        DataFrame con columnas codificadas.
    """
    logger.info("Aplicando reglas de codificación")
    df_transformed = df.copy()
    for col, params in encoding_rules.items():
        if col in df_transformed.columns:
            df_transformed = encode_categorical(df_transformed, col, params.get("drop"))
    return df_transformed


def select_final_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Selecciona columnas finales para el modelo.

    Args:
        df: DataFrame transformado.
        target: Nombre de la columna objetivo.

    Returns:
        DataFrame con columnas seleccionadas.
    """
    logger.info("Seleccionando columnas finales")
    numeric_cols = [
        col for col in df.columns if df[col].dtype in ["int64", "float64"] and col != target
    ]
    encoded_cols = [
        col
        for col in df.columns
        if col.startswith(("Comportamiento_", "Mezcla_", "Pago_Minimo_", "Ocupacion_"))
    ]
    target_col = [target]
    final_cols = list(dict.fromkeys(numeric_cols + encoded_cols + target_col))
    return df[final_cols]


def transform_data(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Transforma datos aplicando codificación y selección de columnas.

    Args:
        df: DataFrame limpio.
        config: Configuración con reglas de transformación.

    Returns:
        DataFrame transformado.
    """
    logger.info("Iniciando transformación de datos")
    df_transformed = df.copy()
    df_transformed = apply_encoding_rules(df_transformed, config.process.encoding)
    df_transformed = select_final_columns(df_transformed, "Puntaje_Credito")
    return df_transformed


def create_feature(df: pd.DataFrame, name: str, formula: dict) -> pd.DataFrame:
    """Crea una nueva feature basada en una fórmula.

    Args:
        df: DataFrame original.
        name: Nombre de la nueva feature.
        formula: Diccionario con operación y columnas involucradas.

    Returns:
        DataFrame con la nueva feature.
    """
    logger.info(f"Creando feature: {name}")
    df_new = df.copy()
    if "Salario_Mensual" in df_new.columns:
        df_new["Salario_Mensual"] = pd.to_numeric(df_new["Salario_Mensual"], errors="coerce")
    try:
        if name == "debt_to_income":
            df_new[name] = df_new["Deuda_Pendiente"] / df_new["Salario_Mensual"]
        elif name == "payment_to_income":
            df_new[name] = df_new["Total_Cuota_Mensual"] / df_new["Salario_Mensual"]
        elif name == "credit_history_ratio":
            df_new[name] = df_new["Edad_Historial_Credito"] / df_new["Edad"]
        df_new[name] = df_new[name].replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception as e:
        logger.error(f"Error al crear feature {name}: {e}")
        df_new[name] = 0
    return df_new


def create_new_features(df: pd.DataFrame, new_features: list) -> pd.DataFrame:
    """Crea múltiples nuevas features según la configuración.

    Args:
        df: DataFrame transformado.
        new_features: Lista de features con nombres y fórmulas.

    Returns:
        DataFrame con nuevas features.
    """
    logger.info("Creando nuevas features")
    df_new = df.copy()
    for feature in new_features:
        df_new = create_feature(df_new, feature.name, feature.formula)
    return df_new


def process_data(config: DictConfig) -> tuple:
    """Procesa datos crudos y genera conjuntos de entrenamiento/prueba.

    Args:
        config: Configuración con rutas y parámetros de procesamiento.

    Returns:
        Tupla con X_train, X_test, y_train, y_test (pd.DataFrame).

    Raises:
        FileNotFoundError: Si el archivo crudo no existe.
    """
    logger.info("Iniciando procesamiento de datos")
    data = get_data(config.raw.path)
    data = clean_data(data, config)
    data = transform_data(data, config)
    data = create_new_features(data, config.process.new_features)
    feature_cols = [col for col in data.columns if col != config.process.target]
    X = data[feature_cols]
    y = data[[config.process.target]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.process.test_size,
        random_state=config.process.random_state,
    )
    output_dir = Path(get_original_cwd()) / config.processed.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / config.processed.X_train.name, index=False)
    X_test.to_csv(output_dir / config.processed.X_test.name, index=False)
    y_train.to_csv(output_dir / config.processed.y_train.name, index=False)
    y_test.to_csv(output_dir / config.processed.y_test.name, index=False)
    logger.info(f"Datos guardados en {output_dir}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """Punto de entrada para ejecutar el procesamiento directamente."""
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(config: DictConfig):
        process_data(config)

    main()