import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

# Load dataset
def load_image_caption_data(image_dir, caption_file):
    with open(caption_file, "r") as file:
        lines = file.readlines()

    image_paths = []
    captions = []

    for line in lines:
        img, caption = line.strip().split('\t')
        image_paths.append(os.path.join(image_dir, img))
        captions.append(caption)

    # Creating a dataset
    data = {'image': image_paths, 'caption': captions}
    dataset = Dataset.from_dict(data)
    return dataset

# Preprocessing function for images and captions
def preprocess_data(examples, processor):
    # Load images
    images = [Image.open(img_path).convert("RGB") for img_path in examples['image']]
    
    # Preprocess images and captions
    inputs = processor(images=images, text=examples['caption'], return_tensors="pt", padding=True)
    return inputs

# Paths
image_dir = "./data/images"
caption_file = "./data/captions.txt"

# Load and preprocess dataset
dataset = load_image_caption_data(image_dir, caption_file)

# Initialize the BLIP Processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Preprocessing function
def process_data(example):
    return preprocess_data(example, processor)

# Apply preprocessing
dataset = dataset.map(process_data, batched=True)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = dataset.train_test_split(test_size=val_size, seed=42).values()

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=200,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=processor,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./models/captioning_model")
processor.save_pretrained("./models/captioning_model")

print("Training complete and model saved!")
