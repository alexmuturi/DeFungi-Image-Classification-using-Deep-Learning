import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL



model_path = "mobilenetv2"  
loaded_model = tf.keras.models.load_model(model_path)

# Define custom class labels

class_labels = [
    "H1: Candida albicans",
    "H2: Aspergillus niger",
    "H3: Trichophyton rubrum",
    "H5: Trichophyton mentagrophytes",
    "H6: Epidermophyton floccosum"
]


#preprocess an image for prediction
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

#make predictions
def make_predictions(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    processed_img = preprocess_image(img)
    predictions = loaded_model.predict(processed_img)

    # Decode predictions using custom labels
    decoded_predictions = [(label, score) for label, score in zip(class_labels, predictions.flatten())]

    # Sort predictions by score in descending order
    decoded_predictions.sort(key=lambda x: x[1], reverse=True)

    return decoded_predictions



# Streamlit app
st.set_page_config(
    page_title="Fungi Classifier",
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.sidebar.title("About")
st.sidebar.info(
    
    ":man: Alex Muturi "
    ":woman: Maryan Hajir"
    ":man: Joe Malombe "
    
)
st.title("Fungi Image Classification 🍄")
"A simple transfer learning image classifier with MobileNetV2 base CNN "
st.write ("Upload an image from the repository for a prediction")


# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")



if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False,width=300  )

    # Make predictions when the user clicks the "Predict" button
    if st.sidebar.button("Predict"):
        predictions = make_predictions(uploaded_file)
        st.subheader("Predictions:")
        for i, (label, score) in enumerate(predictions):
            st.write(f"{i + 1}. {label}: {score:.2%}")