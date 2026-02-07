# Platt Lorrain Training & Deployment

This guide walks you through fine-tuning and deploying a Platt Lorrain chatbot.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Prepare     │ ──▶ │  2. Train       │ ──▶ │  3. Deploy      │
│  Training Data  │     │  (Colab + LoRA) │     │  (Together+HF)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Step 1: Prepare Training Data

Already done! Your training data is at `training/platt_chat_train.jsonl`.

To regenerate or customize:

```bash
python training/prepare_dataset.py \
    --input corpus/output.jsonl \
    --output training/platt_chat_train.jsonl \
    --examples-per-item 2
```

Options:
- `--examples-per-item 3` → More training examples (default: 2)
- `--include-questionable` → Include QUESTIONABLE validations

## Step 2: Fine-Tune with QLoRA

### 2.1 Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Upload `training/platt_finetune.ipynb`
3. Select **Runtime → Change runtime type → T4 GPU**

### 2.2 Upload Training Data

When prompted, upload `training/platt_chat_train.jsonl`

### 2.3 Run All Cells

Training takes ~10-15 minutes on a free T4 GPU.

At the end, you'll download `platt-lorrain-lora.zip` containing your LoRA adapter.

## Step 3: Deploy to Together AI

### 3.1 Create Together AI Account

1. Go to [together.ai](https://together.ai/) and sign up
2. Get your API key from the dashboard

### 3.2 Upload Your Model

**Option A: Use Together AI's fine-tuning API (easiest)**

```bash
pip install together

# Upload your training data and fine-tune directly
together files upload training/platt_chat_train.jsonl
together fine-tuning create \
    --training-file <file-id> \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --suffix platt-lorrain
```

**Option B: Upload LoRA adapter (if you trained in Colab)**

1. Unzip `platt-lorrain-lora.zip`
2. Push to Hugging Face Hub:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/platt-lorrain-lora platt-lorrain-lora/
   ```
3. Use Together AI's Hugging Face integration

### 3.3 Test Your Model

```python
from together import Together

client = Together(api_key="your-api-key")

response = client.chat.completions.create(
    model="your-username/platt-lorrain",  # Your model ID
    messages=[
        {"role": "system", "content": "Du sprichst Platt Lorrain."},
        {"role": "user", "content": "Wie geht es dir?"}
    ]
)
print(response.choices[0].message.content)
```

## Step 4: Deploy Gradio App to Hugging Face Spaces

### 4.1 Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `platt-lorrain-chat`
3. Select **Gradio** as the SDK
4. Choose **Public** visibility

### 4.2 Upload Files

Upload these files to your Space:
- `app.py`
- `requirements.txt`

### 4.3 Add Secrets

In your Space settings, add these secrets:
- `TOGETHER_API_KEY` → Your Together AI API key
- `MODEL_ID` → Your fine-tuned model ID (e.g., `your-username/platt-lorrain`)

### 4.4 Done!

Your app will be live at: `https://huggingface.co/spaces/your-username/platt-lorrain-chat`

## Cost Estimates

| Service | Cost |
|---------|------|
| Colab Training | Free (T4 GPU) |
| Together AI Fine-tuning | ~$2-5 one-time |
| Together AI Inference | ~$0.20/1M tokens |
| Hugging Face Spaces | Free |

For a chatbot with moderate usage (~1000 messages/day), expect ~$1-5/month.

## Alternative: Quick Start Without Fine-Tuning

If you want to test the app immediately without fine-tuning, you can use the base Mistral model with a strong system prompt. The app will work, just less accurately.

1. Deploy `app.py` to HF Spaces
2. Set `MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3`
3. The system prompt will guide the model to attempt Platt

## Files in This Directory

```
training/
├── README.md                  # This file
├── prepare_dataset.py         # Converts corpus to training format
├── platt_chat_train.jsonl     # Generated training data
├── platt_finetune.ipynb       # Colab notebook for training
├── app.py                     # Gradio chat app
└── requirements.txt           # Python dependencies for app
```

## Troubleshooting

**Training runs out of memory in Colab:**
- Use a smaller batch size (change `per_device_train_batch_size=1`)
- Reduce `max_seq_length` to 1024

**Model doesn't speak good Platt:**
- Add more training data to your corpus
- Increase `num_train_epochs` to 5
- Review your corpus for quality issues

**Together AI returns errors:**
- Check your API key is correct
- Verify your model ID matches what's in the dashboard
- Check rate limits on your account
