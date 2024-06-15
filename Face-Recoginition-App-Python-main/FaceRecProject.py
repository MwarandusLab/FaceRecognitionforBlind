import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
import speech_recognition as sr
import time

# Initialize pyttsx3 engine offline voice play sound text to speech
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def process_known_face(name, x1, y1, x2, y2):
    speak("Face found on the database.")
    speak(f"The name of the person is {name}")
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


def process_unknown_face(x1, y1, x2, y2):
    speak("Face not found on the database. Would you like to add this person to the database?")
    # Uncomment the following line if you want to draw a rectangle around the face
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
    # cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


def add_new_person():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Please say the name of the person")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        name = text[text.find("is") + 3: text.find("save")].strip()
        speak(f"The name of the person is {name}. Face will be saved on the database.")
        # Capture face and save it
        face_image = "ImagesAttendance/" + name + ".jpg"  # Path to save the image
        cv2.imwrite(face_image, img)  # Save the whole image
        speak("Face saved on the database.")
        return name
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand what you said.")
        return None
    except sr.RequestError as e:
        speak("Sorry, my speech service is down.")
        return None


path = "ImagesAttendance"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

r = sr.Recognizer()  # Initialize the recognizer outside the loop

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            process_known_face(name, x1, y1, x2, y2)
        else:
            process_unknown_face(x1, y1, x2, y2)
            # Wait for 5 seconds for user response
            time.sleep(5)
            speak("Would you like to add this person to the database?")
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)  # Adjust noise levels before listening
                audio = r.listen(source)
            try:
                response = r.recognize_google(audio).lower()
                if "yes" in response:
                    new_name = add_new_person()
                    if new_name:
                        # Assuming successful addition of new face to the database
                        # Update the encodeListKnown and classNames accordingly
                        encodeListKnown.append(encodeFace)
                        classNames.append(new_name)
                        speak("Database updated. Restarting the system.")
                        # Restart the loop to update the face recognition
                        break
                else:
                    speak("No new face added. Continuing scanning.")
            except sr.UnknownValueError:
                speak("Sorry, I couldn't understand what you said.")
            except sr.RequestError as e:
                speak("Sorry, my speech service is down.")

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
