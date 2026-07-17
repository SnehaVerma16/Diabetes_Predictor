# app.py


import os
import gradio as gr
import joblib

# Load the trained model
deployed_dt = joblib.load("diabetes_prediction_model.pkl")


# Prediction Function
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
):

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]]

    prediction = deployed_dt.predict(input_data)

    if prediction[0] == 1:
        return "⚠️ High Risk of Diabetes\n\nPlease consult a healthcare professional for further evaluation."
    else:
        return "✅ Low Risk of Diabetes\n\nMaintain a healthy lifestyle and regular health checkups."


# Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Slider(0, 17, value=1, step=1, label="👶 Pregnancies"),
        gr.Slider(0, 200, value=120, step=1, label="🩸 Glucose (mg/dL)"),
        gr.Slider(0, 130, value=70, step=1, label="💓 Blood Pressure (mm Hg)"),
        gr.Slider(0, 100, value=20, step=1, label="📏 Skin Thickness (mm)"),
        gr.Slider(0, 900, value=80, step=1, label="💉 Insulin (mu U/ml)"),
        gr.Slider(0.0, 70.0, value=25.0, step=0.1, label="⚖️ BMI"),
        gr.Slider(0.05, 3.0, value=0.50, step=0.01, label="🧬 Diabetes Pedigree Function"),
        gr.Slider(18, 100, value=30, step=1, label="🎂 Age"),
    ],

    outputs=gr.Textbox(
        label="🩺 Prediction Result"
    ),

    title="🩺 Diabetes Prediction System",

    description="""
Enter the patient's health details using the sliders below.
The machine learning model will estimate whether the patient is at a **High Risk** or **Low Risk** of diabetes.
""",

    allow_flagging="never"
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
