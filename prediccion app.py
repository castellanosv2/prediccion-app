from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import numpy as np

# Acceso a la BD
DATABASE_URL = "postgresql://postgres:GWxecCCNwRQhCYWpzlMfPuNJXJkjKTBW@turntable.proxy.rlwy.net:32870/railway"

# Tipo modelo
MODEL_FILENAME = "price_model.joblib"

# App
app = FastAPI()

# Conexion a la BD
def get_connection():
    return psycopg2.connect(DATABASE_URL)

# Entrenando modelo
def train_model():
    conn = get_connection()
    df = pd.read_sql_query("SELECT product_id, product_brand, gender, num_images, primary_color, price FROM productos WHERE price IS NOT NULL", conn)
    conn.close()

    # Features y target
    X = df[['product_brand', 'gender', 'num_images', 'primary_color']]
    y = df['price']

    # One Hot Encoding de las categoricas
    cat_features = ['product_brand', 'gender', 'primary_color']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[cat_features])

    # Concatenar con columnas numericas
    X_num = X[['num_images']].values
    X_final = np.hstack([X_cat, X_num])

    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_final, y)

    # Guardar modelo y encoder
    joblib.dump((model, encoder), MODEL_FILENAME)
    print("Modelo entrenado y guardado!")

if os.path.exists(MODEL_FILENAME):
    model, encoder = joblib.load(MODEL_FILENAME)
    print("Modelo cargado desde disco.")
else:
    train_model()
    model, encoder = joblib.load(MODEL_FILENAME)

# === ENDPOINT: test ===
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Precios Activa"}


@app.get("/predict_price")
def predict_price(product_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT product_brand, gender, num_images, primary_color
    FROM productos
    WHERE product_id = %s
    """
    cursor.execute(query, (product_id,))
    result = cursor.fetchone()
    conn.close()

    if result is None:
        raise HTTPException(status_code=404, detail="Producto no encontrado.")

    product_brand, gender, num_images, primary_color = result

    # Preparar el input
    X_input_cat = encoder.transform([[product_brand, gender, primary_color]])
    X_input_num = np.array([[num_images]])
    X_input_final = np.hstack([X_input_cat, X_input_num])

    # Predecir
    predicted_price = model.predict(X_input_final)[0]

    return {"product_id": product_id, "predicted_price": round(predicted_price, 2)}


@app.get("/product_prediction/{product_id}")
def product_prediction(product_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT product_brand, gender, num_images, primary_color, official_price
    FROM productos
    WHERE product_id = %s
    """
    cursor.execute(query, (product_id,))
    result = cursor.fetchone()
    conn.close()

    if result is None:
        raise HTTPException(status_code=404, detail="Producto no encontrado.")

    product_brand, gender, num_images, primary_color, official_price = result

    # Preparar input para prediccion
    X_input_cat = encoder.transform([[product_brand, gender, primary_color]])
    X_input_num = np.array([[num_images]])
    X_input_final = np.hstack([X_input_cat, X_input_num])

    # Predecir
    predicted_price = model.predict(X_input_final)[0]

    return {
        "product_id": product_id,
        "predicted_price": round(predicted_price, 2),
        "official_price": round(official_price, 2) if official_price is not None else None
    }


class SetPriceRequest(BaseModel):
    product_id: int
    official_price: float


@app.post("/set_official_price")
def set_official_price(request: SetPriceRequest):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    UPDATE productos
    SET official_price = %s
    WHERE product_id = %s
    """
    cursor.execute(query, (request.official_price, request.product_id))
    conn.commit()
    conn.close()

    return {
        "message": "Precio oficial actualizado.",
        "product_id": request.product_id,
        "official_price": round(request.official_price, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)