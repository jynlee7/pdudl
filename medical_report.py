import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import requests
import openai
import os

# Load the trained Pneumonia Detection model (PneumoniaCNN)
class PneumoniaCNN(nn.Module):
    # PneumoniaCNN implementation

class MedicalReportGenerator:
    def __init__(self, pneumonia_model_path, gpt3_api_key):
        self.pneumonia_model = PneumoniaCNN()
        self.pneumonia_model.load_state_dict(torch.load(pneumonia_model_path))
        self.pneumonia_model.eval()

        openai.api_key = gpt3_api_key

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(image).unsqueeze(0)  # Add a batch dimension

    def make_pneumonia_prediction(self, input_image):
        with torch.no_grad():
            output = self.pneumonia_model(input_image)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return "Pneumonia Positive" if predicted_class.item() == 1 else "Pneumonia Negative"

    def generate_medical_report(self, image_source, patient_info):
        try:
            if os.path.exists(image_source):  # Local directory
                image = Image.open(image_source)
            else:  # Online source
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content))
                
            input_image = self.preprocess_image(image)

            # Make a Pneumonia Prediction
            pneumonia_prediction = self.make_pneumonia_prediction(input_image)

            # Formulate Input Prompt for GPT-3
            input_prompt = (
                f"Patient: A {patient_info['age']}-year-old {patient_info['gender']} with a chest X-ray showing {pneumonia_prediction}. "
                f"Symptoms: {patient_info['symptoms']}. Medical History: {patient_info['medical_history']}.\n"
                "Diagnosis: Please provide a comprehensive medical report based on the information provided."
            )

            # Request Medical Report from GPT-3
            response = openai.Completion.create(
                engine="davinci",
                prompt=input_prompt,
                temperature=0.7,
                max_tokens=200
            )

            # Process the GPT-3 Output
            generated_report = response.choices[0].text
            return generated_report

        except Exception as e:
            return f"An error occurred: {e}"

# Example usage
pneumonia_model_path = "path_to_pneumonia_model.pth"
gpt3_api_key = "YOUR_GPT_3_API_KEY"

report_generator = MedicalReportGenerator(pneumonia_model_path, gpt3_api_key)

# Example with a local image
local_image_path = "path_to_local_image.jpg"
patient_info = {
    "age": "35",
    "gender": "male",
    "symptoms": "cough, fever",
    "medical_history": "no significant medical history"
}

generated_report = report_generator.generate_medical_report(local_image_path, patient_info)
print("Generated Medical Report:")
print(generated_report)

# Example with an online image
online_image_url = "URL_TO_ONLINE_IMAGE"
patient_info = {
    "age": "45",
    "gender": "female",
    "symptoms": "shortness of breath, chest pain",
    "medical_history": "hypertension"
}

generated_report = report_generator.generate_medical_report(online_image_url, patient_info)
print("Generated Medical Report:")
print(generated_report)
