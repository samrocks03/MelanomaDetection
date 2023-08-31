import streamlit as st
from pipeline import * 
# from PIL import Image
from PIL import Image
import numpy as np

model = MelanomaDetection()



def display_predicted_image(json_value, image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    for obj in json_value['objectsDetected']:
        label = obj['label']
        confidence = obj['confidence']
        bbox = obj['bbox'][0]

        left, top, right, bottom = bbox
        cv2.rectangle(image_array, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        label_text = f"{label}: {confidence}"
        label_bg_color = (0, 0, 0)
        font_scale = 1.4
        label_text_color = (255, 255, 255)
        font_thickness = 2 

        # Get the size of the label text
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Calculate the position for the label background
        label_bg_left = int(left)
        label_bg_top = int(top)
        label_bg_right = int(left) + label_width
        label_bg_bottom = int(top) + label_height + 5  # Adjust the position

        # Draw the label background
        cv2.rectangle(image_array, (label_bg_left, label_bg_top), (label_bg_right, label_bg_bottom), label_bg_color, -1)

        # Calculate the position for the label text
        text_position = (int(left), int(top) + label_height)

        # Draw the label text
        cv2.putText(image_array, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_text_color, font_thickness)

    st.image(image_array, caption="Predicted Image", width=320)


def styled_subheader(text, color, font_size):
    return f'<h3 style="color: {color}; font-size: {font_size}rem;">{text}</h3>'

def main():
    # Set css
    st.markdown(
        """
        <style>
        .title {
            text-align: left;
            color : orange;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Set app title
    st.markdown("<h1 class='title'>Melanoma Detection<br><br></h1>", unsafe_allow_html=True)

    # Display sample images
    #st.subheader("Sample Images")
    st.markdown(styled_subheader("Sample Images", "red", 1.6), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"folder\\benign_1.jpg", caption="Benign")
    with col2:
        st.image(r"folder\\malignant_1.jpg", caption="Malignant")

    # Upload file container
    #st.subheader("Upload Image")
    st.markdown(styled_subheader("Upload Image", "blue", 1.6), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Perform melanoma detection and display results
    if uploaded_file is not None:
        predicted_json, predicted_image = model.perform_melanoma_detection(uploaded_file)
        #st.subheader("Result")
        st.markdown(styled_subheader("Result", "red", 1.6), unsafe_allow_html=True)
        # Display json output, actual and predicted images
        st.json(predicted_json)
        col1, col2 = st.columns(2)
        with col1:
            #st.subheader("Actual Image")
            # st.markdown("<h6 class='title'>Actual Image</h6>", unsafe_allow_html=True)
            st.image(Image.open(uploaded_file), caption = "Actual Image", width = 320)
        with col2:
            # st.markdown("<h6 class='title'>Predicted Image</h6>", unsafe_allow_html=True)
            # st.image(predicted_image, caption = "Predicted Image", width = 320)
            display_predicted_image(predicted_json,uploaded_file)

if __name__ == "__main__":
    main()

# streamlit run app.py

