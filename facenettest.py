from flask import Flask, json, Response, request,jsonify
from os import path, getcwd, walk
import face
import cv2
import numpy as np
import base64
import io
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
    for root,dirs, files in walk("./data/frk/"):
        personitem = Person()
        personitem.name = root
        for file in files:
            cvframe = cv2.imread(root+"/"+file)
            personImage = PersonImage()
            personImage.image =cvframe
            personImage.name = file
            personitem.personImage.append(personImage)
        personList.append(personitem)
    
    file = open("faruk.txt","w") 
    for personitem in personList:
        print(personitem.name)
        embedding = []
        for peronsimage in personitem.personImage:
            test_face = face_recognition.add_identity_3(peronsimage.image)
            embedding.append(test_face)
            dist2 =np.linalg.norm(test_face - embedding[0])
            print('-----------------')
            print(personitem.name + '#---#' + peronsimage.name + '#---#' + str(dist2))
            file.write(personitem.name + '#---#' + peronsimage.name + '#---#' + str(dist2))
            file.write("\n")
            print('-----------------')
    file.close() 
    
          

if __name__ == '__main__':
    main()