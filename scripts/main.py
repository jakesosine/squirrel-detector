import pandas as pd 
import numpy as np 
import cv2
from PIL import Image, ImageGrab
import os 
import shutil
from roboflow import Roboflow
import pickle 
import base64
import requests
import json


# Ten frame difference between the photos 


def get_img(img_dir):
    return [i for i in os.listdir(img_dir)]


def decide_training(lst_of_imgs, 
                    base_dir = '../images/training_images/'
                    ):
    for image_path in lst_of_imgs:
        img = cv2.imread(base_dir + image_path)
        cv2.imshow('TEST', img)
        right_test = input("Yes to Keep - No to Destroy:")
    pass 
    

def crop_picture(img,
                 x,
                 y,
                 h,
                 w):
    """
    
    """
    cropped = img[y:y+h,x:x+w]
    return cropped, img 

def get_roboflow_model(api_key='t4r4m454mfPHkngCFoFW'):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("squirrel-backyard-science-yes-letsgo")
    model = project.version(1).model
    return model 



def get_squirrel_position():
    pass 

def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_frame_by_frame(model,
                       video_name='Ring_Garden_20230509_0915.mp4',):
    cap = cv2.VideoCapture(base_dir + video_name)
    indx = 0 
    trans_dir = '../images/transition/'
    preds = []
    while cap.isOpened():
        indx += 1
        ret, img = cap.read()
        if not (ret):
            break
        cv2.imwrite(f'{trans_dir}{indx}.jpg', img)
        predictions = model.predict(f'{trans_dir}{indx}.jpg', confidence=40, overlap=10)
        predictions.save(f'{trans_dir}{indx}.jpg')
        predictions_json = predictions.json()
        cv2.imshow('IMG',img)
        for jsn in predictions_json['predictions']:
            d = {**jsn,**{'fname':video_name}}
            preds.append(d)
            print(jsn)
        key = cv2.waitKey(33)
        if key == 27:
            break
    pd.DataFrame(preds).to_csv(f'../data/01_raw/{video_name}.csv')
    cv2.destroyAllWindows()
    import glob
    import re 
    numbers = re.compile(r'(\d+)')
                         
    img_array = []
    for filename in sorted(glob.glob('../images/transition/*.jpg'),key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(f'../videos/output_video/{video_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    [os.remove('../images/transition/' + i) for i in os.listdir('../images/transition/') if i.endswith('.jpg')]

def get_motion(path_to_video,
               video_path,
               contour_threshold=7500,
               ):
    """
    
    """
    cap = cv2.VideoCapture(path_to_video)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    iterator = 0 
    while cap.isOpened():
        if not (ret):
            break
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for indx, contour in enumerate(contours): 
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < contour_threshold:
                continue 
            else: 
                cropped, img = crop_picture(img=frame1,
                                            x=x,
                                            y=y,
                                            h=h,
                                            w=w)
                cv2.imwrite(f'../images/cropped_images/{video_path}--{iterator}-{y}-{x}-{w}-{h}_cropped_image.jpg', cropped)
                cv2.imwrite(f'../images/large_images_no_annotation/{video_path}--{iterator}-{y}-{x}-{w}-{h}_image.jpg', img)

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0), 2)
                # cv2.putText(img, f"Status:".format("Movement"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                cv2.imshow('inter12',frame1[y:y+h,x:x+w])
                cv2.imwrite(f'../images/large_images/{video_path}--{iterator}-{y}-{x}-{w}-{h}_image.jpg', img)
                iterator += 1
        cv2.imshow('inter',frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    


if __name__=='__main__':

    base_dir = '../videos/never_seen/'
    model = pickle.load(open('model.pkl','rb'))
    lst = [i for i in os.listdir(base_dir) if i.endswith('mp4')]
    for i in lst:
        get_frame_by_frame(model,
                           video_name=i)
    # model.predict('../images/never_seen/test.png', confidence=0, overlap=10).save('never_seen.jpg')




    # lst = [i for i in os.listdir('../videos/squirrel/') if i.endswith('mp4')]
    # jpg_lst = [i for i in os.listdir('../images/large_images/') if i.endswith('.jpg')]
    # rf = Roboflow(api_key="t4r4m454mfPHkngCFoFW")
    # project = rf.workspace().project("squirrel-backyard-science")
    # model = project.version(1).model
    # pickle.dump(model, open('model.pkl','wb'))

    # while cap.isOpened():
    #     ret, img = cap.read()
    #     # cv2.imshow('IMG',img)
    #     key = cv2.waitKey(33)
    #     if key == 27:
    #         break
    #     image = infer()
    # # And display the inference results
    #     cv2.imshow('image', image)
    # cv2.destroyAllWindows()
    
    

    # res = model.predict(f'../images/large_images/{jpg_lst[0]}', confidence=40, overlap=30)#.save("prediction.jpg")
    # print(f'Processing {len(lst)} videos')
    # for video in lst:
        # print(video[:video.find('.')])
        # get_motion(f'../videos/squirrel/{video}', video)
