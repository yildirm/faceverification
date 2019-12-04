from flask import Flask, json, Response, request,jsonify
from os import path, getcwd, walk
import face
import cv2
import numpy as np
import base64
import io
import scipy.misc
import ast
from flasgger import swag_from, Swagger
import shutil

class Person:
    def __init__(self):
        self.name = None
        self.personImage =[]

class PersonImage:
    def __init__(self):
        self.image =None
        self.name = None
        self.emembedding =None
        self.disc =None

def main():
    personList =[]
    face_recognition = face.Recognition(min_face_size=20)
    for root,dirs, files in walk("./data/frk"):
        personitem = Person()
        personitem.name = root
        for file in files:
            print(root+"/"+file)
            cvframe = cv2.imread(root+"/"+file)
            personImage = PersonImage()
            personImage.image =cvframe
            personImage.name = file
            personitem.personImage.append(personImage)
        personList.append(personitem)
    
    for personitem in personList:
        for peronsimage in personitem.personImage:
            print('-----------------')
            img =face_recognition.getFace(peronsimage.image)
            if img != -1:
                 cv2.imwrite(peronsimage.name,img[0].image)
                 cv2.imshow('ssdf',img[0].image)
                 cv2.waitKey(0)
                 cv2.destroyAllWindows()
            else:
                print(peronsimage.name)
    
   
          
if __name__ == '__main__':
    main()