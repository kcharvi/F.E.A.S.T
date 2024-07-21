from dotenv import load_dotenv
load_dotenv()

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
import os

from transformers import __all__ as available_classes
print(available_classes)

sys.path.append(os.path.abspath('yolov7'))

model_path ="best.pt"
hugging_face_token="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"

def load_yolo_model():
    from models.experimental import attempt_load
    return attempt_load

###################### RECIPE GENERATION METHODS ######################
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
###################### RECIPE GENERATION METHODS ######################

###################### OBJECT DETCTION METHODS ######################
@st.cache_data
def load_model(model_path):
    device = torch.device('cpu') 
    attempt_load = load_yolo_model()
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
            print("Class ID:", class_id)
            print("Class Confidence:", class_confidence)
            print("Raw Prediction:", prediction)
            if class_confidence > threshold:
                ingredient = class_labels.get(int(class_id), "Unknown")
                print("Detected Ingredient:", ingredient)
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
###################### OBJECT DETCTION METHODS ######################

###################### NUTRITIONAL VALUE ######################
def get_nutritional_values(ingredients):
    hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')
    if not hugging_face_token:
        print("Hugging Face token is missing.")
        return None

    print("Ingredients: ", ingredients)
    url = "https://api-inference.huggingface.co/models/sgarbi/bert-fda-nutrition-ner"
    headers = {"Authorization": f"Bearer {hugging_face_token}"}
    payload = {
        "inputs": ["Here are the ingredients to use: " + ingredients]
    }
    print("Payload: ", payload)

    max_retries = 5
    retry_delay = 20  # seconds

    for attempt in range(max_retries):
        response = requests.post(url, json=payload, headers=headers)
        print("Response Status Code: ", response.status_code)
        print("Response Content: ", response.content)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            print(f"Model is loading. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            try:
                error_detail = response.json()
            except ValueError:
                error_detail = response.content
            print(f"Error: {response.status_code}, {error_detail}")
            return None

    print("Max retries reached. Failed to retrieve nutritional values.")
    return None
###################### NUTRITIONAL VALUE ######################


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("F.E.A.S.T - Food & Ingredient AI Suggestion Technology")
    st.subheader("What's in Your Kitchen? Let's Create a Dish")
    poster_path = "Eyes_on_Eats_Poster.jpg"
    st.image(poster_path, caption="EYES ON EATS", use_column_width=True)
    
    st.write("MODEL: Object Detection using YOLO")

    model = load_model(model_path)
    class_labels = load_class_labels("Final_classes.txt")
    upload_option = st.radio("Upload Image Option:", ("Upload from File", "Capture from Camera", "Try from Sample Images"))

    ingredients = []  # List to store detected ingredients
    
    if upload_option == "Upload from File":
        uploaded_file = st.file_uploader("Upload an image of the ingredients", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                predictions = detect_ingredients(image, model)
                new_ingredients = parse_predictions(predictions, class_labels)
                st.write("Predicted Ingredients:", new_ingredients)
                ingredients += new_ingredients  # Append new ingredients to the existing list

                ingredients_string = ", ".join(new_ingredients)

                generate_recipe_button = st.button("Generate Recipe", key="generate_recipe")
                if generate_recipe_button:
                    st.write("Loading Recipe...")
                    generated_recipes = generation_function(ingredients_string)
                    st.header("Generated Recipe:")
                    for recipe_text in generated_recipes:
                        print_recipe(recipe_text)
                    st.write("----------------------------------------------------------------------------------------------------------------------------------")
                
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

    elif upload_option == "Try from Sample Images":
        sample_image_path = "sample_image_1.jpg"  # Update with the correct path to your sample image
        try:
            image = Image.open(sample_image_path)
            st.image(image, caption="Sample Image", use_column_width=True)
            predictions = detect_ingredients(image, model)
            new_ingredients = parse_predictions(predictions, class_labels)
            st.write("Predicted Ingredients:", new_ingredients)
            ingredients += new_ingredients  # Append new ingredients to the existing list

            ingredients_string = ", ".join(new_ingredients)

            generate_recipe_button = st.button("Generate Recipe", key="generate_recipe")
            if generate_recipe_button:
                st.write("Loading Recipe...")
                generated_recipes = generation_function(ingredients_string)
                st.header("Generated Recipe:")
                for recipe_text in generated_recipes:
                    print_recipe(recipe_text)
                st.write("----------------------------------------------------------------------------------------------------------------------------------")
            
            nutrition_recipe_button = st.button("Get Nutritional Content", key="nutritional_value")
            if nutrition_recipe_button:
                nutritional_info = get_nutritional_values(", ".join(new_ingredients))
                if nutritional_info is not None:
                    st.write("Nutritional Values:")
                    st.write(nutritional_info)
                else:
                    st.error("Failed to retrieve nutritional values.")
        
        except Exception as e:
            st.error(f"Error processing the sample image: {str(e)}")

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