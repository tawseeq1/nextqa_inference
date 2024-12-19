import json
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import pandas as pd

# Path configurations
data_dir = "NExT-QA/dataset/nextqa"
video_dir = "path_to_videos"
qa_file = os.path.join(data_dir, "val.csv")

map_file = os.path.join(data_dir, "map_vid_vidorID.json")

# Load Chameleon 7B model and tokenizer
model_name = "models"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto")
    
model.eval()  
# Ensure GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load QA annotations
qa_data = pd.read_csv(qa_file)

# Load video ID to file mapping
with open(map_file, "r") as f:
    vid_mapping = json.load(f)

# Prepare dataset
qa_pairs = []
for _, row in qa_data.iterrows():
    video_id = row["video"]
    if video_id in vid_mapping:
        video_path = os.path.join(video_dir, f"{vid_mapping[video_id]}.mp4")
        qa_pairs.append({
            "video_path": video_path,
            "question": row["question"],
            "answer": row["answer"],
            "options": [row[f"a{i}"] for i in range(5)]
        })

# Inference function
def infer(video_path, question, options):
    input_text = f"Question: {question}\nOptions: {', '.join(options)}\nVideo: {video_path}\nAnswer:"
    inputs = processor(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Run inference
results = []
for pair in qa_pairs:
    video_path = pair["video_path"]
    question = pair["question"]
    options = pair["options"]
    predicted_answer = infer(video_path, question, options)
    results.append({
        "video": pair["video_path"],
        "question": question,
        "predicted_answer": predicted_answer,
        "true_answer": pair["answer"]
    })

# Save results
output_file = "inference_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Inference completed. Results saved to {output_file}.")