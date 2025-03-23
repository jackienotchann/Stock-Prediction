import streamlit as st
import pickle
import pandas as pd

# Load trained models from .pkl files
def load_models():
    try:
        model = pickle.load(open(r'C:\Users\mohit\OneDrive\Desktop\Stock Prediction\model.pkl', 'rb'))
        knn_model = pickle.load(open(r'C:\Users\mohit\OneDrive\Desktop\Stock Prediction\knn_model.pkl', 'rb'))
        bagging_model = pickle.load(open(r'C:\Users\mohit\OneDrive\Desktop\Stock Prediction\bagging_model.pkl', 'rb'))
        gb_model = pickle.load(open(r'C:\Users\mohit\OneDrive\Desktop\Stock Prediction\gb_model.pkl', 'rb'))
        adaboost_model = pickle.load(open(r'C:\Users\mohit\OneDrive\Desktop\Stock Prediction\adaboost_model.pkl', 'rb'))
        print("Models loaded successfully!")
        return model, knn_model, bagging_model, gb_model, adaboost_model
    except Exception as e:
        print("Error loading models:", e)
        return None, None, None, None, None


# Main function to run the Streamlit app
def main():
    # Set page title
    st.title("Stock Prediction Web App")
    
    # Load trained models
    model, knn_model, bagging_model, gb_model, adaboost_model = load_models()
    
    # Display a message
    st.write("Models loaded successfully!")
    
    # Allow user input for prediction
    st.write("Enter values for prediction:")
    
    # Add UI components for user input (e.g., sliders, dropdowns, text inputs)
    adj_close = st.slider("Adj Close", min_value=0.0, max_value=1000.0, step=0.1)
    high = st.slider("High", min_value=0.0, max_value=1000.0, step=0.1)
    low = st.slider("Low", min_value=0.0, max_value=1000.0, step=0.1)
    open_value = st.slider("Open", min_value=0.0, max_value=1000.0, step=0.1)
    volume = st.slider("Volume", min_value=0.0, max_value=1000000.0, step=1000.0)
    
    # Allow user input for start and end dates
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    # Make predictions based on user input
    if st.button("Predict"):
        # Prepare input data as DataFrame
        input_data = pd.DataFrame({
            'Adj Close': [adj_close],
            'High': [high],
            'Low': [low],
            'Open': [open_value],
            'Volume': [volume]
        })
        
        # Make predictions using each model
        lstm_rnn_prediction = model.predict(input_data)
        knn_prediction = knn_model.predict(input_data)
        bagging_prediction = bagging_model.predict(input_data)
        gradient_boosting_prediction = gb_model.predict(input_data)
        adaboost_prediction = adaboost_model.predict(input_data)
        
        # Display predictions
        st.write("LSTM and RNN Combined Prediction:", lstm_rnn_prediction)
        st.write("KNN Prediction:", knn_prediction)
        st.write("Bagging Prediction:", bagging_prediction)
        st.write("Gradient Boosting Prediction:", gradient_boosting_prediction)
        st.write("AdaBoost Prediction:", adaboost_prediction)

# Run the main function
if __name__ == "__main__":
    main()
