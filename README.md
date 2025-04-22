# 🤖 Cyber-GPT2: Fine-Tuned Conversational GPT-2

This project demonstrates how to fine-tune and use a conversational version of **GPT-2** using a custom dataset of questions and answers, followed by testing the model interactively.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `cyber_gpt2.py` | Loads a dataset of Q&A pairs and fine-tunes the GPT-2 model using HuggingFace's Transformers library. |
| `cyber_gpt2_testmodel.py` | Loads the fine-tuned model and generates dynamic responses interactively using user input. |

---

## 🧠 Model Features

- Based on HuggingFace’s `GPT-2`
- Trained on a custom Q&A dataset
- Uses `[END]` token to define conversation boundaries
- Dynamic and iterative response generation
- Trained with `Trainer` and `DataCollatorForLanguageModeling`

---

## ⚙️ Dependencies

Install the required packages via pip:

```bash
pip install transformers torch pandas tqdm
```

---

## 🚀 How to Use

### Step 1: Train the model

Make sure your dataset is in CSV format and contains `questions` and `answers` columns. Then run:

```bash
python cyber_gpt2.py
```

This will fine-tune the GPT-2 model and save it in the `cyber_model/` directory.

### Step 2: Test the model

After training is complete, run:

```bash
python cyber_gpt2_testmodel.py
```

This will launch an interactive chat loop where you can test your model.

---

## 📌 Notes

- Make sure the dataset path is properly set in `cyber_gpt2.py`
- You can customize generation parameters (e.g., temperature, top_k, repetition_penalty)
- The model dynamically adjusts `max_length` until it hits `[END]` or a safety limit

---

## 👨‍💻 Author

Amirali Kamali  
[GitHub: AmiraliKamali](https://github.com/AmiraliKamali)
