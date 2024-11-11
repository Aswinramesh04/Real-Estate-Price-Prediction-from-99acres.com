import streamlit as st 
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # For encoding the location

# Load the trained model (ensure the correct path to your pickle file)
with open('regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Encoding the location feature
le = LabelEncoder()

def predict_price(location, plot_sqft, Bedroom, Bathroom):
    x = np.zeros(4)  # Assuming 4 features: location, plot_sqft, Bedroom, Bathroom
    
    x[0] = location  # Location should be numeric (encoded)
    x[1] = plot_sqft
    x[2] = Bedroom
    x[3] = Bathroom
    

    return model.predict([x])[0]

# Streamlit app
st.title('🏡 House Price Prediction')
st.write('Fill in the details below to predict the price of a house.')
df = pd.read_csv("Mumbai_Property (1).csv")

# Input fields for user to enter data
st.sidebar.header("User Input Features")

# Encode the location
df['Region'] = le.fit_transform(df['Region'])  # Label encode the regions
location_names = dict(zip(le.classes_, range(len(le.classes_))))  # To display the region names
location = st.sidebar.selectbox('Select the Location', (df['Region'].sort_values().unique()), format_func=lambda x: le.inverse_transform([x])[0])

plot_sqft = st.sidebar.slider("Select Total Area in SqFt", 500, int(max(df['Plot_area'])), step=100)
bathroom = st.sidebar.selectbox("Enter Number of Bathroom", (df['Bathroom'].sort_values().unique()))
Bedroom = st.sidebar.selectbox("Enter Number of Bedroom", (df['Bedroom'].sort_values().unique()))

# Predict price
if st.button("Calculate Price"):
    result = predict_price(location, plot_sqft, Bedroom, bathroom)
    st.success(f'Total Price in Lakhs: ₹{result:.2f} Lakhs')

# Show information about the model or data
st.sidebar.info('This prediction model is based on historical data of house prices, factoring in region, plot area, rate per square foot, and number of bedrooms.')

# Additional styling to make it more creative
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    ---
    **Developed by Loga Aswin**  
    """)



# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load the trained model (ensure the correct path to your pickle file)
# with open('regression_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# def predict_price(plot_sqft, Bedroom):
#     x = np.zeros(7)
    
#     x[0] = plot_sqft
#     x[1] = Bedroom

#     return model.predict([x])[0]

# st.title('🏡 House Price Prediction')
# st.write('Fill in the details below to predict the price of a house.')
# df = pd.read_csv("Mumbai_Property (1).csv")

# # Input fields for user to enter data
# st.sidebar.header("User Input Features")
# location = st.sidebar.selectbox('Select the Location',(df['Region'].sort_values().unique()))
# plot_sqft = st.sidebar.slider("Select Total Area in SqFt",500,int(max(df['Plot_area'])),step=100)
# bathroom = st.sidebar.selectbox("Enter Number of Bathroom",(df['Bathroom'].sort_values().unique()))
# Bedroom = st.sidebar.selectbox("Enter Number of Bedroom",(df['Bedroom'].sort_values().unique()))

# # Store the inputs into a numpy array for model prediction
# input_data = np.array([[location, plot_sqft, Bedroom, bathroom]])

# if st.button("Calculate Price"):
#         result = predict_price(plot_sqft,Bedroom)
#     st.success('Total Price in Lakhs : {}'.format(result))
# # # Predict the house price
# # if st.sidebar.button('Predict Price'):
# #     prediction = model.predict(input_data)
# #     st.success(f'🏠 The predicted house price is **₹{prediction[0]:,.2f} Lakh**')

# # Show information about the model or data
# st.sidebar.info('This prediction model is based on historical data of house prices, factoring in region, plot area, rate per square foot, and number of bedrooms.')

# # Additional styling to make it more creative
# st.markdown("""
#     <style>
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 8px;
#         padding: 10px 20px;
#     }
#     .stButton button:hover {
#         background-color: #45a049;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Footer
# st.markdown("""
#     ---
#     **Developed by Loga Aswin**  
#     """)
