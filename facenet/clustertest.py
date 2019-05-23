# MIT License
#
# Copyright (c) 2017 PXL University College
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import argparse
import cv2
#import facenet
from si_facenet.src import facenet
from  si_facenet.src.align import detect_face
import facenet.src.align.detect_face as align_detect_face
from sklearn.cluster import DBSCAN
import time


def add_overlays(frame, faces, frame_rate,dist):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if dist < 0.8:
                cv2.putText(frame, 'mehmet', (face_bb[0], face_bb[3]),
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


    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    model_path = find_model_in_folder()
    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret == 0:
            print("Error")
            return

        #faces = face_recognition.identify(frame)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_path)
                #image_list = load_images_from_folder("./data/images/")
                image_list_test = load_images_from_folder("./data/test/")

                face_crop_size=160
                face_crop_margin=32
                min_face_size=20
                pnet, rnet, onet = Detection._setup_mtcnn()

                image = Detection.find_faces(image_list_test[0])
                cv2.imshow("test",images[10])
                cv2.waitKey(0) & 0xFF
            

                imagesframe = []
                imagesframe.append(frame)
                
                images = align_data(imagesframe, args.image_size, args.margin, pnet, rnet, onet)
                images_test = align_data(image_list_test, args.image_size, args.margin, pnet, rnet, onet)
            
                #cv2.imshow("test",images[10])
                #cv2.waitKey(0) & 0xFF
                print(images)
                embcode = embeddingimage(images,sess)
                embcodetest = embeddingimage(images_test,sess)
            
                disst = ecuclidean_distance_2(embcode,embcodetest[0])
            
                if (frame_count % frame_interval) == 0:
                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count / (end_time - start_time))
                        start_time = time.time()
                        frame_count = 0
                    
                new_frame = add_overlays(frame.copy(), imagesframe, frame_rate,disst)
                frame_count += 1
                cv2.imshow("Face Recognition", new_frame)
            
            keyPressed = cv2.waitKey(1) & 0xFF
            if keyPressed == 27: 
                break

    video_capture.release()
    cv2.destroyAllWindows()

   
            

           
def ecuclidean_distance_2(emb_list, embedding):
        for emb in emb_list:
            dist2 =np.linalg.norm(embedding-emb)
            print('  %1.4f  ' % dist2, end='')
            print("\n")
            return dist2

def find_model_in_folder():
		for root, dirs, files in os.walk("./"):
			for file in files:
				if file.endswith(".pb"):
					return os.path.join(root, file)

def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    print(len(image_list))

    for x in range(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


def embeddingimage(images,sess):
     images_placeholder = sess.graph.get_tensor_by_name("input:0")
     embeddings = sess.graph.get_tensor_by_name("embeddings:0")
     phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

     prewhiten_face = facenet.prewhiten(images)
     feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
     emb = sess.run(embeddings, feed_dict=feed_dict)
     return emb


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def ecuclidean_distance(emb, embtest):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb, embtest))))

    dist2 =np.linalg.norm(embtest-emb)
    print('dist değeri')
    print(dist)
 


    print('dist 2 nin değer')
    print(dist2)
    if dist2 < 0.52 :
        print('test başarılı')

    #print('  %1.4f  ' % dist, end='')
    print("\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=1)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--largest_cluster_only', action='store_true',
                        help='This argument will make that only the biggest cluster is saved.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)

class Detection:
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32, min_face_size=20):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = min_face_size

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align_detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align_detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
