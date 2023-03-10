from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Diabetes(BaseModel):
    Pregnancies : int
    Glucose	: int
    BloodPressure : int
    SkinThickness : int
    Insulin	: int
    BMI	: float
    DiabetesPedigreeFunction : float
    Age : int	