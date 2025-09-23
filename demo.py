import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from model import LeNet5
import streamlit as st

st.title("📝👀 Yann Lecun's First Rudimentary ZipCode Reader - First Computer Vision System in History 🤔")

# load LeNet5 - 0.99 percent accuracy
model = LeNet5() # instantiating the LeNet5() class
model.load_state_dict(torch.load('models/LeNet5_wts.pth'))
model.eval() # enabling inference mode

# enabling GPU access
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
decoded_digits = []

# Upload image
uploaded_file = st.file_uploader("Upload Captured ZipCode", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to grayscale
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold (binary inverse)
    _, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = sorted(rects, key=lambda x: x[0])  # sort left-to-right

    output_img = img.copy()
    digit_rois = []

    for rect in rects:
        cv2.rectangle(output_img, (rect[0], rect[1]), 
                      (rect[0] + rect[2], rect[1] + rect[3]), 
                      (0, 255, 0), 2)
    plt.imshow(output_img)
    plt.axis('off')
    st.pyplot(plt)
    for rect in rects:
        cv2.rectangle(output_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
        #plt.imshow(output_img)
        #plt.axis("off")

        # output_img: original image
        # co-ordinates returned by cv1.boundingRect(): (x-topleft, y_topleft, length, width)
        # (rect[0], rect[1]): co-ordinates of the top left 
        # (rect[0] + rect[2], rect[1] + rect[3]): co-ordinates of the bottom right
        # (0, 255, 0): green
        # 1: thickness

        # region of interest extraction
        leng = int(rect[3] * 1.6) # more room
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2 - 2) # finding vertical center and offsetting it to center of image
        pt2 = int(rect[0] + rect[2] //2 - leng // 2 - 2) # finding horiziontal center and offsetting it to the center of the image
        roi = im_th[pt1:pt1 + leng + 2, pt2:pt2 + leng + 2] # extracting a square box around the horiziontal and vertical center of the digit with an offset of 2 for more room
        roi = cv2.resize(roi, (28, 28)).astype(np.float32) / 255.0  # resize to match LeNet5 input shape & normalize
        roi_tensor = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0) # roi tensor for model submisson
        roi_tensor = roi_tensor.to(device)
        with torch.no_grad():
            output = model(roi_tensor) # perform inference
            predicted_label = torch.argmax(output, dim = 1).item() # get label
            decoded_digits.append(predicted_label) # push to labels list
    
decoded_str = " ".join(str(x) for x in decoded_digits)
st.markdown(f"### 🔢 Zip Code: **{decoded_str}**")
