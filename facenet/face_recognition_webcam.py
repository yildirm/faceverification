import argparse
import sys
import time

import cv2
import os
import face
import datetime
import facenet



def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.dist is not None:
                if face.dist < 0.7:
                    cv2.putText(frame, "Mehmet YILDIRIM", (face_bb[0], face_bb[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
                if face.dist > 0.7:
                    cv2.putText(frame, "Dogrulanmadi", (face_bb[0], face_bb[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)


    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    return frame


def main(args):
    frame_interval = 30  
    fps_display_interval = 5 
    frame_rate = 0
    frame_count = 0

    if args.debug:
        face.debug = True

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    face_recognition = face.Recognition(min_face_size=20)

    image_list_test = load_images_from_folder("./data/test/")
    test_face = face_recognition.add_identity(image_list_test[0],'test_1')
    print(test_face.embedding)
    start_time = time.time()
    print('add identity eklendi..')

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret == 0:
            print("Error")
            return

        faces = face_recognition.identify(frame,test_face)
        #facecrop =face_recognition.detect.find_faces(frame)
        #cv2.imshow("Face Recognition", faces)

        if (frame_count % frame_interval) == 0:
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        new_frame = add_overlays(frame.copy(), faces, frame_rate)
        frame_count += 1
        cv2.imshow("Face Recognition", new_frame)

        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27: 
            break

    video_capture.release()
    cv2.destroyAllWindows()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
