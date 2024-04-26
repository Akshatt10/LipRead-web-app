import streamlit as st
import os
import numpy as np
import tempfile
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')
with st.sidebar:
    st.image("C:\\Users\\Akshat tyagi\\Downloads\\LipRead.png")
    st.title("Lip Interpretation")
    st.info("Developed to perform lip reading task over a given video")

st.title("Video Platform")
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options:  
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        input_filepath = os.path.join('..', 'data', 's1', selected_video)
        output_filepath = 'converted_video.mp4'

    # Convert selected video to mp4 using FFmpeg
        os.system(f'ffmpeg -i "{input_filepath}" -vcodec libx264 "{output_filepath}" -y')

    # Display converted video
        video_bytes = open(output_filepath, 'rb').read()
        st.video(video_bytes)


    with col2: 
        st.info('Extracted part of video for ML model to predict')
        video, annotations = load_data(tf.convert_to_tensor(input_filepath))
        
        DEMO_VIDEO = r"C:\Users\Akshat tyagi\Downloads\animation-ezgif.com-video-to-mp4-converter.mp4"
        tfflie = tempfile.NamedTemporaryFile(delete=False)
        tfflie.name = DEMO_VIDEO
        st.video(tfflie.name)
        
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        

        
        
        