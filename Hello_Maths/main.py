import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(layout="wide")

st.image('Hello.png')
col1, col2 = st.columns([4, 4])

with col1:
    run = st.checkbox('Process', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Response")
    output_text_area = st.subheader("")

# Configure the AI model
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (0, 255, 0), 15)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this math problem", pil_image])
    # Extracting text from the response and cleaning unnecessary characters
    text = response.candidates[0].content.parts[0].text
    return text.replace('$\\boxed{', '').replace('}$', '')

prev_pos = None
canvas = None
output_text = ""

# Continuously get frames from the webcam
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        if fingers == [1, 1, 1, 1, 1]:
            output_text = sendToAI(model, canvas)
            print(output_text)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)
