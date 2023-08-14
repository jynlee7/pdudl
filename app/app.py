from flask import Flask, request, jsonify
from MedicalReportGenerator import MedicalReportGenerator

app = Flask(__name__)

pneumonia_model_path = "path_to_pneumonia_model.pth"
gpt3_api_key = "YOUR_GPT_3_API_KEY"
report_generator = MedicalReportGenerator(pneumonia_model_path, gpt3_api_key)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        image = request.files['image']
        age = request.form['age']
        gender = request.form['gender']
        symptoms = request.form['symptoms']
        medical_history = request.form['medical_history']
        
        # Process image, generate report
        report = report_generator.generate_medical_report(image, {
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "medical_history": medical_history
        })

        return report

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run()
