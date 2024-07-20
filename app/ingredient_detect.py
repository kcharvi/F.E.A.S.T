import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer
import time
import requests

sys.path.append("F:\\Desktop\\DL\\Project\\bhanucha_charviku_final_project\\bhanucha_charviku_final\\application\\yolov7") 
from models.experimental import attempt_load 

model_path ="F:\\Desktop\\DL\\Project\\bhanucha_charviku_final_project\\bhanucha_charviku_final\\application\\best.pt"
API_TOKEN="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"

# RECIPE GENERATION METHODS
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)
prefix = "items: "

generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)
    return new_texts

@st.cache_data
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    print(inputs)
    inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True, return_tensors="jax")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
    generated = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=False)
    generated_recipe = target_postprocessing(generated, special_tokens)
    return generated_recipe

def print_recipe(recipe_text):
    sections = recipe_text.split("\n")
    for section in sections:
        section = section.strip()
        if section.startswith("title:"):
            section = section.replace("title:", "").strip()
            headline = "TITLE"
        elif section.startswith("ingredients:"):
            section = section.replace("ingredients:", "").strip()
            headline = "INGREDIENTS"
        elif section.startswith("directions:"):
            section = section.replace("directions:", "").strip()
            headline = "DIRECTIONS"
        
        if headline == "TITLE":
            st.write(f"[{headline}]: {section.capitalize()}")
        else:
            section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
            st.write(f"[{headline}]:")
            for info in section_info:
                st.write(info)
    
    st.write("-" * 130)
# RECIPE GENERATION METHODS

# OBJECT DETCTION METHODS
@st.cache_data
def load_model(model_path):
    device = torch.device('cpu') 
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model

def detect_ingredients(_image, _model):
    image = np.array(_image)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = _model(image)
        predictions = outputs[0]
        print("Predictions Shape:", predictions.shape)
    return predictions

def parse_predictions(predictions, class_labels, threshold=0.1):
    predictions = predictions.squeeze(0)  
    detected_ingredients = []

    for prediction in predictions:
        conf = prediction[4]  
        if conf > threshold:
            class_probs = prediction[5:]  
            class_id = class_probs.argmax()  
            class_confidence = class_probs[class_id]
            # print("Class ID:", class_id)
            # print("Class Confidence:", class_confidence)
            # print("Raw Prediction:", prediction)
            if class_confidence > threshold:
                ingredient = class_labels.get(int(class_id), "Unknown")
                # print("Detected Ingredient:", ingredient)
                if ingredient not in detected_ingredients:
                    detected_ingredients.append(ingredient)

    return detected_ingredients

@st.cache_data
def load_class_labels(filepath):
    class_labels = {}
    with open(filepath, 'r') as file:
        for idx, line in enumerate(file):
            class_labels[idx] = line.strip()
    return class_labels
# OBJECT DETCTION METHODS

# NUTRITIONAL VALUE
def get_nutritional_values(ingredients):
    print("INGred: ", ingredients)
    url = "https://api-inference.huggingface.co/models/sgarbi/bert-fda-nutrition-ner"
    headers = {"Authorization": "Bearer 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"}
    payload = {
        "inputs": ingredients
    }
    print("PAYLoad: ",payload)
    response = requests.post(url, json=payload, headers=headers)
    print("RESPONSE: ",response)

    if response.status_code == 200:
        return response.json()
    else:
        print()
        return None
#NUTRITIONAL VALUE


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("CSE 676: Deep Learning")
    st.header("EYES ON EATS: From Image to Formula")
    st.header("Snap, Cook, Enjoy: Satisfy Your Cravings!")
    st.write("What's in Your Kitchen? Let's Create a Dish")
    st.write("MODEL: Object Detection using YOLO")

    model = load_model(model_path)
    class_labels = load_class_labels("F:\\Desktop\\DL\\Project\\bhanucha_charviku_final_project\\bhanucha_charviku_final\\application\\Final_classes.txt")
    upload_option = st.radio("Upload Image Option:", ("Upload from File", "Capture from Camera"))

    ingredients = []  # List to store detected ingredients
    
    if upload_option == "Upload from File":
        uploaded_file = st.file_uploader("Upload an image of the ingredients", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                predictions = detect_ingredients(image, model)
                new_ingredients = parse_predictions(predictions, class_labels)
                print("detected ingredients: ", new_ingredients)
                st.write("Predicted Ingredients:", new_ingredients)
                print("NEW INGREDIENTS:", new_ingredients)
                ingredients += new_ingredients  # Append new ingredients to the existing list
                print("INGREDIENTS STORED: ", ingredients)

                ingredients_string = ", ".join(new_ingredients)

                generate_recipe_button = st.button("Generate Recipe", key="generate_recipe")
                if generate_recipe_button:
                    st.write("Loading Recipe...")
                    generated_recipes = generation_function(ingredients_string)
                    st.header("Generated Recipe:")
                    for recipe_text in generated_recipes:
                        print_recipe(recipe_text)
                    st.write("----------------------------------------------------------------------------------------------------------------------------------")
                
                # Retrieve nutritional values
                nutrition_recipe_button = st.button("Get Nutritional Content", key="nutritional_value")
                if nutrition_recipe_button:
                    nutritional_info = get_nutritional_values(", ".join(new_ingredients))
                    if nutritional_info is not None:
                        st.write("Nutritional Values:")
                        st.write(nutritional_info)
                    else:
                        st.error("Failed to retrieve nutritional values.")
            
                
            except Exception as e:
                st.error(f"Error processing the recipe: {str(e)}")
    else:
        st.write("Please allow access to your camera.")
        camera = cv2.VideoCapture(0)
        if st.button("Capture Image"):
            st.write("Get ready to take a snap!")
            time.sleep(3)  # Add a delay of 3 seconds
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            predictions = detect_ingredients(image, model)
            new_ingredients = parse_predictions(predictions, class_labels)

            st.write("Predicted Ingredients:", new_ingredients)
            ingredients += new_ingredients  # Append new ingredients to the existing list
            
            generate_recipe_button = st.button("Generate Recipe", key="generate_recipe")
            if generate_recipe_button:
                st.write("Loading Recipe...")
                recipe = generation_function(ingredients)
                st.header("Generated Recipe:")
                st.write(recipe)
            
    st.markdown("**Made by: Tarun and Charvi**")  

if __name__ == '__main__':
    main()
