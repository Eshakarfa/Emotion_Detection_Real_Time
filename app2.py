import tkinter as tk
from tkinter import Button, Label
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def Detect(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            
           
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            
            cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            
            label1.config(text=pred)
    except:
        pass

def detect_from_webcam():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        Detect(frame)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Delay to allow GUI events to process
        top.update_idletasks()
        top.update()
    cap.release()
    cv2.destroyAllWindows()

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json", "model_weights1.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

upload = Button(top, text="Use Webcam", command=detect_from_webcam, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
