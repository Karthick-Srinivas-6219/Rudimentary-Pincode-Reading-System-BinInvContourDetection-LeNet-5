# 📨🤖Rudimentary handwritten pincode reading system using Binary Inverse Thresholding based Contour Detection & LeNet-5 🤔
This project is an attempt to replicate the world's first computer vision system. In the 80s the US Postal Service required a system to automatically recognize handwritten Zip Codes to sort mail more efficiently. Yann LeCun et. al. took on this challenge &amp; eventually ended up proposing LeNet-5. 

# Demo 👇
<video src="demo.mp4" controls width="640"></video>
[[Link to Demo]](https://youtu.be/VscNUvCFYMw "Click to watch")

# Overview of the pipeline
![Alt text](zipcode_reader.png)

## 🚀 Features

* **Binary Inverse Image Thresholding**: A critical intensity transform that represents the handwritten digits with **intensity value 0** and the background with **intensity value 1** hence amplifying the **intensity gradient**.
* **OpenCV Contour Detection**: Records **coordinates** of the regions in the image where there is a **steep gradient rise from 0 to 255** and applies bounding boxes involving those coordinates.
* **LeNet-5 trained on the MNIST dataset**: The downstream model which **recognizes the digits** whose region bounding boxes were clipped.
---

## 📂 Project Structure

```bash
.
├── bbt_gallery/              # Registered gallery of actors from the show - "The Big Bang Theory".
├── bbt_test_images/          # Some test scenes from the show - "The Big Bang Theory" to be supplied for inference.
├── office_gallery/           # Registered gallery of actors from the show - "The Office".
├── office_test_images/       # Some test scenes from the show - "The Office" to be supplied for inference.
├── models/                   # Models which are used in the extraction of face embeddings
       ├── dlib_face_recognition_resnet_model_v1.dat
       ├── mmod_human_face_detector.dat
├── requirements.txt      # Python dependencies.
├── get_gallery_embeddings.ipynb     # To generate embeddings for the pre-registered gallery of actors.
├── crop_recog_persistant_inf.ipynb  # Runs the entire inference pipeline i.e. supply test image --> faces get detected and cropped --> Embeddings get generated and matched with the cached gallery embeddings.
├── demo.py            # A Streamlit demo of the entire project.
├── my_dlib_funcs.py   # Some utility functions for embeddings generation and caching.
├── gallery_embeddings.pkl   # Embedding cache represented as a pickle file.
├── requirements.txt   # Project dependencies
├── dlib-19.24.99-cp313-cp313-win_amd64.whl  # dlib wheel for python3.13
```
