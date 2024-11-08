import streamlit as st
import pandas as pd
import pickle
import re
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score

ps = PorterStemmer()

def clean_and_stem_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = ' '.join(ps.stem(word) for word in text.split())  # Stemming
    return text

# Define sidebar and pages
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Upload and Train", "Single/Batch Prediction", "About"])

# Load pre-trained model components if they exist
try:
    tfidf = joblib.load("tfidf_transformer.pkl")
    model = joblib.load("Trained_Model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except:
    tfidf, model, label_encoder = None, None, None

def predict_comment_category(comment):
    cleaned_comment = clean_and_stem_text(comment)  # Clean and stem input
    X_test = tfidf.transform([cleaned_comment])  # Transform text with TF-IDF
    y_pred = model.predict(X_test)  # Predict category
    predicted_category = label_encoder.inverse_transform(y_pred)  # Decode label
    return predicted_category[0]

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>Paradox 📹</h1>", unsafe_allow_html=True)
    st.image("Youtube image.png", use_column_width=True)

    st.markdown("""
        **Paradox** is your smart assistant for categorizing YouTube comments.
        ### 🌟 How Paradox Works:
        1. **Upload Training Data**: Navigate to **Upload and Train** to provide data in the required format for model training.
        2. **Single or Batch Prediction**: Go to **Single/Batch Prediction** for predictions on individual or multiple comments.
    """)

# Upload and Train Page
elif app_mode == "Upload and Train":
    st.markdown("<h1 style='text-align: center;'>Upload and Train 🔄</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your training CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        if 'Comment' in data.columns and 'Label' in data.columns:
            data['Comment'] = data['Comment'].apply(clean_and_stem_text)
            
            X = data['Comment']
            y = data['Label']
            
            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            # TF-IDF transformation
            tfidf = TfidfVectorizer(max_features=5000)
            X_tfidf = tfidf.fit_transform(X)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Training completed with accuracy: {accuracy:.2f}")
            
            # Save model components
            joblib.dump(tfidf, "tfidf_transformer.pkl")
            joblib.dump(model, "Trained_Model.pkl")
            joblib.dump(label_encoder, "label_encoder.pkl")
            
            # Download links for saved models
            st.download_button("Download TF-IDF Transformer", data=open("tfidf_transformer.pkl", "rb").read(), file_name="tfidf_transformer.pkl")
            st.download_button("Download Trained Model", data=open("Trained_Model.pkl", "rb").read(), file_name="Trained_Model.pkl")
            st.download_button("Download Label Encoder", data=open("label_encoder.pkl", "rb").read(), file_name="label_encoder.pkl")
        else:
            st.error("Uploaded file must contain 'Comment' and 'Label' columns.")

# Single/Batch Prediction Page
elif app_mode == "Single/Batch Prediction":
    st.markdown("<h1 style='text-align: center;'>Single/Batch Prediction 📝</h1>", unsafe_allow_html=True)
    
    if model is None or tfidf is None or label_encoder is None:
        st.error("Please train a model on the 'Upload and Train' page before making predictions.")
    else:
        # Single comment prediction
        user_comment = st.text_area("Enter a single YouTube comment to classify:")
        
        if st.button("Predict Category 🔍"):
            if user_comment:
                start_time = time.time()
                predicted_category = predict_comment_category(user_comment)
                end_time = time.time()
                
                st.write(f"**Predicted Category**: {predicted_category}")
                st.write(f"**Time taken for prediction**: {end_time - start_time:.2f} seconds")
            else:
                st.error("Please enter a comment to predict.")

        st.markdown("---")
        
        # Batch prediction
        uploaded_predict_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")
        
        if uploaded_predict_file is not None:
            predict_data = pd.read_csv(uploaded_predict_file)
            
            if 'Comment' in predict_data.columns:
                predict_data['Comment'] = predict_data['Comment'].apply(clean_and_stem_text)
                X_predict = tfidf.transform(predict_data['Comment'])
                y_pred = model.predict(X_predict)
                
                # Decode predictions
                predictions = label_encoder.inverse_transform(y_pred)
                predict_data['Label'] = predictions
                
                # Save and download predictions
                result_file = "batch_predictions.csv"
                predict_data[['Comment', 'Label']].to_csv(result_file, index=False)
                st.download_button("Download Predictions", data=open(result_file, "rb").read(), file_name=result_file)
                
                st.success("Predictions completed and available for download.")
            else:
                st.error("Uploaded file must contain a 'Comment' column.")

# About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center;'>About</h1>", unsafe_allow_html=True)
    st.markdown("""
        **Paradox** was developed to help content creators categorize and analyze YouTube comments.
        - **Future Improvements**: Expanding the dataset, accuracy improvements, and adding sentiment analysis.
        - **Creators**: Mohd Adnan Khan and Muhammed Ashrah.
    """)
