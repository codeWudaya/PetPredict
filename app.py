from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

def main():
    # Set title and description for your app
    st.title('Cat or Dog Classifier')
    st.sidebar.title('Options')
    st.sidebar.markdown('Upload an image and click the "Classify" button.')

    # Create file upload functionality
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        img = Image.open(uploaded_file)
        img = img.resize((64, 64))  # Resize the image if necessary
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize pixel values (if necessary)
        img_array = img_array / 255.0

        # Load the saved model
        model = tf.keras.models.load_model('model.h5')

        # Make predictions
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            result = 'Dog'
        else:
            result = 'Cat'

        # Display the image and prediction result
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write('Prediction:', result)

if __name__ == '__main__':
    main()