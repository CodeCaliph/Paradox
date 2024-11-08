import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
import time
import joblib


tfidf = joblib.load("tfidf_transformer.pkl")
model = joblib.load("Trained_Model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
    
ps = PorterStemmer() 

def clean_and_stem_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = ' '.join(ps.stem(word) for word in text.split())  # Stemming
    return text

def predict_comment_category(comment):
    cleaned_comment = clean_and_stem_text(comment)  # Clean and stem input
    X_test = tfidf.transform([cleaned_comment])  # Transform text with TF-IDF
    y_pred = model.predict(X_test)  # Predict category
    predicted_category = label_encoder.inverse_transform(y_pred)  # Decode label
    return predicted_category[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Comment Classification"])

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
Paradox was created to help content creators organize and analyze YouTube comments more effectively, making it easier to manage audience feedback and insights.""")

# Comment Classification Page
elif app_mode == "Comment Classification":
    st.markdown("<h1 style='text-align: center;'>Comments Classification 📝</h1>", unsafe_allow_html=True)

    # Input field for comment
    user_comment = st.text_area("Enter a YouTube comment to classify:")

    # Predict button
    if st.button("Predict Category 🔍"):
        if user_comment:
            start_time = time.time()

            # Spinner for loading
            with st.spinner("Analyzing the comment...🧠"):
                time.sleep(1)
                st.success("Prediction complete! ✅")

            # Make prediction using provided comment
            predicted_category = predict_comment_category(user_comment)
            end_time = time.time()

            # Display Prediction and Confidence Score
            st.subheader("Prediction Result")
            st.write(f"**Predicted Category**: {predicted_category}")
            st.write(f"**Time taken for prediction**: {end_time - start_time:.2f} seconds")
        else:
            st.error("Please enter a comment before predicting. ❌")

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