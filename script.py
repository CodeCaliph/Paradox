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

st.set_page_config(page_title='Paradox', 
                   page_icon="🔮")

ps = PorterStemmer()

def clean_and_stem_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = ' '.join(ps.stem(word) for word in text.split())  # Stemming
    return text

# Define sidebar and pages
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Comments Prediction", "About"])

tfidf = joblib.load("tfidf_transformer.pkl")
model = joblib.load("Trained_Model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

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
        # Welcome to Paradox 📝

        **Paradox** is your smart assistant for analyzing and categorizing YouTube comments into 3 categories namely 'Doubt', 'Feedback' and 'Irrelevant'. With machine learning, Paradox can help you understand the nature of comments on your content. Just enter a comment, and see the predicted category instantly!

        ## 🌟 How Paradox Works:
        1. **Enter a Comment** 💬: Navigate to the **Comment Classification** page and input a YouTube comment.
        2. **Advanced Analysis** 🧠: Paradox will analyze the comment using machine learning.
        3. **Instant Results** 📝: Get a prediction for the comment category instantly.

        ## 🚀 Get Started
        Begin by selecting **Comment Classification** in the sidebar and enter the comment you’d like to classify.

        ## ℹ️ About Us
        Learn more on the **About** page regarding the model, dataset, and the creator behind Paradox.
    """)

# Prediction Page
# Prediction Page
elif app_mode == "Comments Prediction":
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

        # Batch prediction - this is now indented under the Comments Prediction block
        uploaded_predict_file = st.file_uploader("Upload file for batch predictions", type=["csv", "xlsx"])

        if uploaded_predict_file is not None:
            # Load the file based on its extension
            if uploaded_predict_file.name.endswith('.csv'):
                predict_data = pd.read_csv(uploaded_predict_file)
            elif uploaded_predict_file.name.endswith('.xlsx'):
                predict_data = pd.read_excel(uploaded_predict_file)

            # Check for required columns
            if list(predict_data.columns[:2]) == ["ID", "Comment"]:
                # Apply text cleaning and prediction
                predict_data['Comment'] = predict_data['Comment'].apply(clean_and_stem_text)
                X_predict = tfidf.transform(predict_data['Comment'])
                y_pred = model.predict(X_predict)

                # Decode predictions
                predictions = label_encoder.inverse_transform(y_pred)
                predict_data['Label'] = predictions

                # Save and download predictions
                result_file = "batch_predictions.csv"
                predict_data[['ID', 'Comment', 'Label']].to_csv(result_file, index=False)
                st.download_button("Download Predictions", data=open(result_file, "rb").read(), file_name=result_file)

                st.success("Predictions completed and available for download.")
            else:
                st.error("Uploaded file must contain 'ID' and 'Comment' columns as the first two columns.")

# About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center;'>About</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### 📊 About the Dataset
**Paradox** was trained on a large dataset of over 210,000 YouTube comments. These comments are categorized into three groups: **Doubt**, **Irrelevant**, and **Feedback**. This model classifies comment text into these categories to provide insightful organization and understanding of audience responses.

---

### 👤 About the Creators

**Mohd Adnan Khan**  
- **Background**: A passionate data science professional specializing in data science, machine learning, and deep learning. Adnan is committed to developing intelligent solutions that simplify complex challenges in AI.
- **Contact**: [mohdadnankhan.india@gmail.com](mailto:mohdadnankhan.india@gmail.com) | [LinkedIn](https://www.linkedin.com/in/mohd-adnan--khan)  

**Muhammed Ashrah**  
- **Role**: Data Science Collaborator  
- **Background**: A dedicated data science enthusiast with a strong interest in machine learning and AI. He is passionate about building intuitive and user-friendly tools.
- **Contact**: [mohd.ashrah@gmail.com](mailto:mohd.ashrah@gmail.com) | [LinkedIn](https://www.linkedin.com/in/muhammed-ashrah)

---

### 🔗 Connect with Us
Feel free to reach out for inquiries, collaborations, or to learn more about our work in machine learning!

---

### 🔮 Future Improvements
- Expanding the dataset to include more comment categories
- Increasing accuracy and processing efficiency
- Integrating sentiment analysis and other advanced features

---

### 💡 Project Motivation
Paradox was created to help content creators organize and analyze YouTube comments more effectively, making it easier to manage audience feedback and insights.
""")

# Sidebar Information
st.sidebar.subheader("About Paradox 📝")
st.sidebar.text("Classify comments instantly.\nOrganize audience feedback.")

st.sidebar.markdown("Go to **Classification** to start.")
st.sidebar.markdown("---")

st.sidebar.subheader("Key Features")
st.sidebar.text("• Accurate\n• Real-time\n• Easy-to-use")

st.sidebar.markdown("---")
st.sidebar.subheader("Contact")
st.sidebar.markdown("[Email](mailto:mohdadnankhan.india@gmail.com)")
