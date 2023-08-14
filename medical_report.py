import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import requests
import openai

# Load the trained Pneumonia Detection model (PneumoniaCNN)
class PneumoniaCNN(nn.Module):
    # PneumoniaCNN implementation

# Define a function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Add a batch dimension

# Function to generate a medical report
def generate_medical_report(image_url, patient_info, api_key):
    try:
        # Load the pre-trained Pneumonia Detection model
        pneumonia_model = PneumoniaCNN()
        pneumonia_model.load_state_dict(torch.load("path_to_pneumonia_model.pth"))
        pneumonia_model.eval()

        # Set up the GPT-3 API key
        openai.api_key = api_key

        # Load and preprocess the chest X-ray image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        input_image = preprocess_image(image)

        # Make a Pneumonia Prediction using the trained model
        with torch.no_grad():
            output = pneumonia_model(input_image)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        pneumonia_prediction = "Pneumonia Positive" if predicted_class.item() == 1 else "Pneumonia Negative"

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
image_url = "URL_TO_YOUR_CHEST_XRAY_IMAGE"
patient_info = {
    "age": "35",
    "gender": "male",
    "symptoms": "cough, fever",
    "medical_history": "no significant medical history"
}
api_key = "YOUR_GPT_3_API_KEY"

generated_report = generate_medical_report(image_url, patient_info, api_key)
print("Generated Medical Report:")
print(generated_report)
