"""
The source of implementation code:
1. chatgpt: https://platform.openai.com/docs/quickstart  model version: gpt-4o
2. claude 3.5 sonnet: https://github.com/anthropics/anthropic-cookbook   model version: claude-3-5-sonnet-20240620
3. MINICPM V 2_6: https://huggingface.co/openbmb/MiniCPM-V-2_6
4. Biomed CLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
5. Llama 3.2: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct  model version: meta-llama/Llama-3.2-11B-Vision-Instruct
"""
"""
https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset
将COVID, bacterial pneumonia and viral pneumonia混合
"""

import os
import base64
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import anthropic
import google.generativeai as genai
from pathlib import Path
import time
import sys
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoModel, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
import requests


def load_api_key(model_type):
    """Load API key based on the model type."""
    
    if model_type == 'chatgpt':
        return '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'
    elif model_type == 'claude':
        return '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'
    elif model_type == 'gemini':
        return 'AIzaSyC9coNffUwWQNQNE9zxn-uJlsuwv9_H4gU'
    elif model_type in ['biomedclip', 'minicpm', 'llama']:
        return None  # BiomedCLIP and MiniCPM don't require an API key
    else:
        raise ValueError("Unsupported model type. Supported types are: chatgpt, claude, gemini, biomedclip")


def encode_image(image_path):
    """Encode the image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_predictions_to_csv(predictions, output_csv):
    """Save the predictions and ground truth to a CSV file."""
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image Name', 'Ground Truth', 'Prediction'])
        for image_name, ground_truth, prediction in predictions:
            csvwriter.writerow([image_name, ground_truth, prediction])  
            

def evaluate_images(model_key, model_version, image_path, role, content, is_baseline=True):
    """Evaluate images for both augmented and baseline scenarios."""
    api_key = load_api_key(model_key)

    # Initialize the appropriate client or model based on model_key
    if model_key == 'chatgpt':
        client = OpenAI(api_key=api_key)
    elif model_key == 'claude':
        client = anthropic.Anthropic(api_key=api_key)
    elif model_key == 'gemini':
        genai.configure(api_key=api_key, transport='rest')
        generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 2048
        }
        model = genai.GenerativeModel(model_name=model_version, generation_config=generation_config)
    elif model_key == 'biomedclip':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
        model.eval()
    elif model_key == 'minicpm':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    
    elif model_key == 'llama':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MllamaForConditionalGeneration.from_pretrained(
            model_version,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_version)

    results = {}
    confusion_matrices = {}
    labels = ["Normal", "lung diseases"]
    all_predictions = []

    if is_baseline:
        levels = ['baseline']
    else:
        levels = ['strong', 'weak']

    for level in levels:
        results[level] = {'lung diseases': {'tp': 0, 'fn': 0}, 'Normal': {'tn': 0, 'fp': 0}}
        level_predictions = []
        level_true_labels = []

        if is_baseline:
            level_path = image_path
        else:
            level_path = os.path.join(image_path, level)

        if not os.path.exists(level_path):
            print(f"Path does not exist: {level_path}")
            continue

        print(f"Processing level: {level}")

        for class_folder in os.listdir(level_path):
            class_path = os.path.join(level_path, class_folder)
            if not os.path.isdir(class_path) or class_folder not in labels:
                continue

            print(f"Processing class: {class_folder}")

            for image_name in os.listdir(class_path):
                image_file_path = os.path.join(class_path, image_name)
                
                try:
                    if model_key == 'chatgpt':
                        prediction = evaluate_image_chatgpt(client, model_version, image_file_path, role, content)
                    elif model_key == 'claude':
                        prediction = evaluate_image_claude(client, model_version, image_file_path, role, content)
                    elif model_key == 'gemini':
                        prediction = evaluate_image_gemini(model, image_file_path, role, content)
                    elif model_key == 'biomedclip':
                        prediction = evaluate_image_biomedclip(preprocess, model, tokenizer, image_file_path, labels, device)
                    elif model_key == 'minicpm':
                        prediction = evaluate_image_minicpm(model, tokenizer, image_file_path, role, content, device)
                    elif model_key == 'llama':
                        prediction = evaluate_image_llama(model, processor, image_file_path, content, device)
                except Exception as e:
                    print(f"Error processing image {image_file_path}: {e}")
                    continue

                print(f"Image: {image_name}, Ground Truth: {class_folder}, Prediction: {prediction}")

                # Save the original prediction
                all_predictions.append((image_name, class_folder, prediction))

                # Standardize the prediction for metrics calculation
                if "lung diseases" in prediction.lower():
                    standardized_prediction = "lung diseases"
                elif "normal" in prediction.lower():
                    standardized_prediction = "Normal"
                else:
                    standardized_prediction = "unknown"  # or handle this case as appropriate

                level_predictions.append(standardized_prediction)
                level_true_labels.append(class_folder)

                if class_folder == "lung diseases":
                    if standardized_prediction == "lung diseases":
                        results[level]['lung diseases']['tp'] += 1
                    else:
                        results[level]['lung diseases']['fn'] += 1
                else:  # normal
                    if standardized_prediction == "Normal":
                        results[level]['Normal']['tn'] += 1
                    else:
                        results[level]['Normal']['fp'] += 1

        # Calculate metrics
        tp = results[level]['lung diseases']['tp']
        fn = results[level]['lung diseases']['fn']
        tn = results[level]['Normal']['tn']
        fp = results[level]['Normal']['fp']

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        results[level]['metrics'] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy
        }

        confusion_matrices[level] = confusion_matrix(level_true_labels, level_predictions, labels=labels)

    task_name = os.path.basename(image_path)
    return results, confusion_matrices, labels, task_name, all_predictions


def evaluate_image_chatgpt(client, model, image_path, role, content, max_retries=2):
    """Evaluate the image using ChatGPT with retry logic."""
    base64_image = encode_image(image_path)
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred with ChatGPT: {e}. Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                raise Exception(f"Failed to evaluate image with ChatGPT after {max_retries + 1} attempts: {image_path}")

def evaluate_image_claude(client, model, image_path, role, content, max_retries=2):
    """Evaluate the image using Claude with retry logic."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    for attempt in range(max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=role,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": content},
                        ],
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred with Claude: {e}. Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                raise Exception(f"Failed to evaluate image with Claude after {max_retries + 1} attempts: {image_path}")

def evaluate_image_gemini(model, image_path, role, content, max_retries=2):
    """Evaluate the image using Gemini with retry logic."""
    image_path = Path(image_path)
    image_part = {
        "mime_type": "image/png",
        "data": image_path.read_bytes()
    }
    prompt_parts = [
        role + "\n" + content,
        image_part
    ]

    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt_parts)
            if not response.candidates:
                print(response)
                raise Exception("No candidates in the response")
            if not response.candidates[0].content.parts:
                print(response)
                raise Exception("No content parts in the first candidate")
            
            if not response.candidates[0].content.parts[0].text:
                print(response)
                raise Exception("Empty text in the first content part")
           
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred with Gemini: {e}. Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                raise Exception(f"Failed to evaluate image with Gemini after {max_retries + 1} attempts: {image_path}")
    
def evaluate_image_biomedclip(preprocess, model, tokenizer, image_path, labels, device, context_length=256):
    """Evaluate the image using BiomedCLIP."""
    template = 'the diagnosis of this X-rays image is:'
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    
    texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
    
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, texts)
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        
    prediction_index = logits.argmax().item()
    return labels[prediction_index]


def evaluate_image_minicpm(model, tokenizer, image_path, role, content, device, max_retries=2):
    """Evaluate the image using MiniCPM-V-2_6 with retry logic."""
    image = Image.open(image_path).convert('RGB')
    question = content
    msgs = [{'role': 'user', 'content': question}]

    for attempt in range(max_retries + 1):
        try:
            res = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                temperature=0.01
            )
            return res
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred with MiniCPM: {e}. Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                raise Exception(f"Failed to evaluate image with MiniCPM after {max_retries + 1} attempts: {image_path}")

def evaluate_image_llama(model, processor, image_path, content, device, max_retries=2):
    """Evaluate the image using Llama with retry logic."""
    image = Image.open(image_path).convert('RGB')
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": content}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    for attempt in range(max_retries + 1):
        try:
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device)
            output = model.generate(**inputs, max_new_tokens=30)
            return processor.decode(output[0])
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred with Llama: {e}. Retrying... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                raise Exception(f"Failed to evaluate image with Llama after {max_retries + 1} attempts: {image_path}")


def save_confusion_matrix(confusion_matrix, labels, output_image):
    """Save the confusion matrix as an image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_image)
    plt.close()

def save_results_to_csv(results, task_name, output_csv):
    """Save the evaluation results to a CSV file."""
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model Type', 'Task', 'Level', 'Sensitivity', 'Specificity', 'Accuracy'])
        
        for row in results:
            csvwriter.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VLM models on medical images.')
    parser.add_argument('--image_path', type=str, default='/Users/a86153/Desktop/vlm/pneumonia augmented/Rotate',
                        help='Path to the image directory')
    parser.add_argument('--output_prefix', type=str, default='upgrade_claude_x-rays_COT_Rotate',
                        help='Prefix for output files')
    parser.add_argument('--model_key', type=str, default='claude',
                        choices=['chatgpt', 'claude', 'gemini', 'biomedclip', 'minicpm', 'llama'],
                        help='Model to use for evaluation')
    parser.add_argument('--model_version', type=str, default='claude-3-5-sonnet-20241022',#claude-3-5-sonnet-20240620
                        help='Version of the model to use')
    parser.add_argument('--is_baseline', default=False,
                        help='Whether to evaluate baseline scenario')
    args = parser.parse_args()

    role = "Medical knowledge educator"
    content = "How can we identify the features of COVID, bacterial pneumonia and viral pneumonia in a chest X-rays image of a person? Imagine you are an educator tasked with helping a student identify the features of a X-rays image and whether or not the image shows signs of these lung diseases as described by those features. As an educator, conclude your answer in - 'lung diseases' or 'normal'. Describe your reasoning in steps."

    results, confusion_matrices, labels, task_name, all_predictions = evaluate_images(
        args.model_key, args.model_version, args.image_path, role, content, args.is_baseline)
    
    # Save all predictions to CSV
    save_predictions_to_csv(all_predictions, f"{args.output_prefix}_all_predictions.csv")

    # Prepare data for CSV
    csv_data = []
    for level, level_results in results.items():
        if 'metrics' in level_results:
            metrics = level_results['metrics']
            csv_data.append((args.model_key, task_name, level, metrics['sensitivity'], metrics['specificity'], metrics['accuracy']))

    # Save results to CSV
    save_results_to_csv(csv_data, task_name, f"{args.output_prefix}_VLM_evaluation_results.csv")

    # Save confusion matrices
    for level, cm in confusion_matrices.items():
        save_confusion_matrix(cm, labels, f"{args.output_prefix}_confusion_matrix_{level}.png")

    print("Evaluation completed. Results saved.")