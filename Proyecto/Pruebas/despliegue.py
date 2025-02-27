import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo preentrenado
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# Descongelar las últimas 4 capas + capa final (fc)
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:  
        param.requires_grad = True

# Reemplazar la última capa de clasificación
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# Cargar los pesos guardados
model_path = "C:\\Users\\melis\\Desktop\\Proyecto_Cocoa\\Proyecto\\Pruebas\\modelo_cacao_resnet50_finetuned_1.pth"  # Asegúrate de que el archivo esté en la misma carpeta
model.load_state_dict(torch.load(model_path, map_location=device))

# Enviar a GPU si está disponible
model = model.to(device)
model.eval()

print("Modelo cargado exitosamente.")


# Clases del modelo (ajusta según tu dataset)
class_names = ["images_C1", "images_C2", "images_C3", "images_C4"]

# Definir las transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajusta según tu modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización estándar
])

# Función para hacer predicciones
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Preprocesar imagen
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Obtener probabilidades
        predicted_class = torch.argmax(probabilities).item()
    
    return class_names[predicted_class], probabilities.cpu().numpy()

# Construcción de la interfaz en Streamlit
st.title("Clasificador de imágenes de cacao")
st.write("Sube una imagen y el modelo clasificará la fase de maduración.")

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Hacer predicción
    label, probabilities = predict_image(image)

    # Mostrar resultados
    st.subheader(f" Predicción: **{label}**")
    st.write("Probabilidades por clase:")
    
    for i, class_name in enumerate(class_names):
        st.write(f"🔹 {class_name}: {probabilities[i]*100:.2f}%")
