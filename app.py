import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Streamlit app title and description
st.title("Abusive Text Classifier")
st.write("Enter a text below to classify it as 'abusive' or 'non-abusive' using a fine-tuned BanglaBERT model.")

# Step 1: Load the tokenizer and model (cached to avoid reloading on every run)
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "csebuetnlp/banglabert"  # Original pretrained model
    checkpoint_path = "ashrafulparan/BanglaBERT-abusive"  # Hugging Face checkpoint

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model from the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    # Move model to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# Step 2: Initialize the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(["non-abusive", "abusive"])  # Label mapping

# Step 3: Define the prediction function
def predict_single_text(text, model, tokenizer, device):
    model.eval()
    # Tokenize the single text input
    encodings = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Get probabilities
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]   # Get predicted class
    return probs, pred

# Step 4: Create a text input field
text_input = st.text_input("Enter your text here:", "This is a sample text")

# Step 5: Button to trigger prediction
if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text to classify!")
    else:
        # Perform inference
        probs, pred = predict_single_text(text_input, model, tokenizer, device)

        # Decode the prediction
        predicted_label = label_encoder.inverse_transform([pred])[0]
        prob_abusive, prob_non_abusive = probs  # Adjust based on your model's output order

        # Display results
        st.write(f"**Text:** {text_input}")
        st.write(f"**Predicted Label:** {predicted_label}")
        st.write(f"**Probabilities:** Non-abusive: {prob_non_abusive:.4f}, Abusive: {prob_abusive:.4f}")

        # Optional: Add a visual indicator
        if predicted_label == "abusive":
            st.error("This text is classified as abusive!")
        else:
            st.success("This text is classified as non-abusive!")

# Step 6: Footer (optional)
st.markdown("---")
st.write("Powered by BanglaBERT and Streamlit | Model hosted on Hugging Face")