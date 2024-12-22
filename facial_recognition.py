import cv2
import face_recognition

#loading known face encodings:
known_face_encodings = []
known_face_names = []

#load known face and their names here:
known_person1_img = face_recognition.load_image_file(r"C:\Users\rishi\Pictures\Screenshot 2024-12-21 182611.png")
known_person2_img = face_recognition.load_image_file(r"C:\Users\rishi\Pictures\Screenshot 2024-12-21 182655.png")
known_person3_img = face_recognition.load_image_file(r"C:\Users\rishi\Pictures\Screenshot 2024-12-21 182742.png")

known_person1_encoding = face_recognition.face_encodings(known_person1_img)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_img)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_img)[0]

known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)

known_face_names.append("sharukh khan")
known_face_names.append("john abhrahm")
known_face_names.append("hritik roshan")

video_capture = cv2.VideoCapture(0)

while True:
    ret,frame = video_capture.read()

    #finding all the face locations:
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations)

    #loop through each img:
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        #see if any existing face matches:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            #draw a box around the face:
            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        
    cv2.imshow("video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()