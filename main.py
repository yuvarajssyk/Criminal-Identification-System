from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from camera2 import VideoCamera2
from camera3 import VideoCamera3
from camera4 import VideoCamera4
from camera5 import VideoCamera5
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="criminal_face"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()

    ff11=open("sms.txt","w")
    ff11.write("1")
    ff11.close()

    ff=open("facest.txt","w")
    ff.write("")
    ff.close()

    ff=open("sms.txt","w")
    ff.write("0")
    ff.close()

    ff=open("static/frames2.txt","w")
    ff.write("")
    ff.close()

    ff=open("static/frames.txt","w")
    ff.write("")
    ff.close()

    ff=open("static/fstatus.txt","w")
    ff.write("")
    ff.close()

    vid=1
    act=1
    mycursor = mydb.cursor()
    #mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid,act ))
    #value = mycursor.fetchall()

    camera=""
    mycursor.execute("SELECT * FROM cf_camera order by rand()")
    cdd = mycursor.fetchall()
    for cdd1 in cdd:
        camera=cdd1[1]

    ff=open("camera.txt","w")
    ff.write(camera)
    ff.close()

    
    return render_template('web/index.html',msg=msg,act=act)

@app.route('/login_dept', methods=['POST','GET'])
def login_dept():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cf_policestation where station_id=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            session['username'] = username1
            return redirect(url_for('home')) 
        else:
            msg="Your are logged in fail!!!"
                
    
    return render_template('login_dept.html',msg=msg)

@app.route('/login', methods=['POST','GET'])
def login():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cf_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            session['username'] = username1
            return redirect(url_for('admin')) 
        else:
            msg="Your are logged in fail!!!"
                
    
    return render_template('login.html',msg=msg)

@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("SELECT * FROM cf_policestation where id=%s",(did,))
        dd = mycursor.fetchone()
        sid=dd[6]

        mycursor.execute("delete from cf_camera where station_id=%s",(sid,))
        mydb.commit()
    
        mycursor.execute("delete from cf_policestation where id=%s",(did,))
        mydb.commit()
        msg="ok"
   
    return render_template('admin.html',msg=msg,data=data)



@app.route('/add_station',methods=['POST','GET'])
def add_station():
    msg=""
    act=request.args.get("act")
    email=""
    mess=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM cf_policestation")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1

    v1=str(maxid)  
    ss=v1.zfill(3)
    sid="S"+ss

    cm=v1.zfill(2)
    camera="Camera"+cm

    
    if request.method=='POST':
        station_id=request.form['station_id']
        station_name=request.form['station_name']
        mobile=request.form['mobile']
        email=request.form['email']
        area=request.form['area']
        city=request.form['city']
        
    
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")

        rn=randint(10000,99999)
        pass1=str(rn)

        mycursor.execute("SELECT count(*) FROM cf_policestation where station_id=%s or mobile=%s or email=%s",(station_id,mobile,email))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            sql = "INSERT INTO cf_policestation(id,station_name,mobile,email,area,city,station_id,password) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,station_name,mobile,email,area,city,station_id,pass1)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
            mess="Police Station: "+station_name+", Police Station ID:"+station_id+", Password"+pass1

            ##
            rn1=randint(0,4)
            rn2=randint(5,9)
            camera1="Camera"+str(maxid)+str(rn1)
            camera2="Camera"+str(maxid)+str(rn2)
            
            mycursor.execute("SELECT max(id)+1 FROM cf_camera")
            maxid2 = mycursor.fetchone()[0]
            if maxid2 is None:
                maxid2=1
            maxid3=maxid2+1
            sql = "INSERT INTO cf_camera(id,camera,station_id) VALUES (%s, %s, %s)"
            val = (maxid2,camera1,station_id)            
            mycursor.execute(sql, val)
            mydb.commit()

            sql = "INSERT INTO cf_camera(id,camera,station_id) VALUES (%s, %s, %s)"
            val = (maxid3,camera2,station_id)            
            mycursor.execute(sql, val)
            mydb.commit()
            ##
            msg="success"
        else:
            msg="fail"


    mycursor.execute("SELECT * FROM cf_policestation")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cf_policestation where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_station'))
        
   
    return render_template('add_station.html',msg=msg,mess=mess,email=email,data=data,act=act,sid=sid)

@app.route('/home',methods=['POST','GET'])
def home():
    msg=""
    data=[]
    act=request.args.get("act")
    uname=""
    ff=open("facest.txt","w")
    ff.write("")
    ff.close()
    
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM cf_police where station_id=%s",(uname,))
    data1 = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cf_police where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('home'))

        
    return render_template('home.html',msg=msg,act=act,data=data,data1=data1)

@app.route('/add_police',methods=['POST','GET'])
def add_police():
    msg=""
    act=request.args.get("act")
    sid=request.args.get("sid")
    email=""
    mess=""
    data=[]
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where id=%s",(sid,))
    data = mycursor.fetchone()
    station_id=data[6]

    mycursor.execute("SELECT max(id)+1 FROM cf_police")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1

    v1=str(maxid)  
    pp=v1.zfill(3)
    pid="P"+pp

    
    if request.method=='POST':
        police_id=request.form['police_id']
        police_name=request.form['police_name']
        mobile=request.form['mobile']
        email=request.form['email']
        
        mycursor.execute("SELECT count(*) FROM cf_police where police_id=%s or mobile=%s or email=%s",(police_id,mobile,email))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
        
            sql = "INSERT INTO cf_police(id,station_id,police_name,mobile,email,police_id) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (maxid,station_id,police_name,mobile,email,police_id)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
          
            msg="success"
        else:
            msg="fail"

    mycursor.execute("SELECT * FROM cf_police where station_id=%s",(station_id,))
    data1 = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cf_police where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_police',sid=sid))
    
   
    return render_template('add_police.html',msg=msg,mess=mess,email=email,data=data,data1=data1,act=act,pid=pid,sid=sid)

@app.route('/entry',methods=['POST','GET'])
def entry():
    msg=""
    act=request.args.get("act")
    email=""
    mess=""
    data=[]
    entryby=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()


    if request.method=='POST':
        police_id=request.form['police_id']
        mycursor.execute("SELECT count(*) FROM cf_police where police_id=%s && station_id=%s",(police_id,uname))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            entryby=police_id
            msg="ok"
        else:
            msg="fail"

    return render_template('entry.html',msg=msg,data=data,act=act,entryby=entryby)
        
@app.route('/add_case',methods=['POST','GET'])
def add_case():
    msg=""
    vid=""
    act=request.args.get("act")
    entryby=request.args.get("entryby")
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")

    mycursor = mydb.cursor()
    if request.method=='POST':
        name=request.form['name']
        gender=request.form['gender']
        dob=request.form['dob']
        address=request.form['address']
        complaint=request.form['complaint']
        complainant_name=request.form['complainant_name']
        complainant_address=request.form['complainant_address']
        place=request.form['place']
        
        complaint_date=request.form['complaint_date']
        
        district=request.form['district']
        fir_date=request.form['fir_date']
        jail_period=request.form['jail_period']
        release_date=request.form['release_date']
        
        police_inspector=request.form['police_inspector']

        mycursor.execute("SELECT max(id)+1 FROM cf_criminal_details")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        fnn=""
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fn=file.filename
            fnn="P"+str(maxid)+fn  
            #fn1 = secure_filename(fn)
            file.save(os.path.join("static/upload", fnn))
                
        
        sql = "INSERT INTO cf_criminal_details(id,name,gender,dob,address,complaint,complainant_name,complainant_address,place,complaint_date,district,fir_date,jail_period,release_date,proof,police_station,police_inspector,entryby,register_date) VALUES (%s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s,%s,%s,%s)"
        val = (maxid,name,gender,dob,address,complaint,complainant_name,complainant_address,place,complaint_date,district,fir_date,jail_period,release_date,fnn,uname,police_inspector,entryby,rdate)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        vid=str(maxid)
        msg="ok"
        

    return render_template('add_case.html',msg=msg,act=act,data=data,vid=vid)


def getImagesAndLabels(path):

    
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    uname=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()
    
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    #ff2=open("mask.txt","w")
    #ff2.write("face")
    #ff2.close()
    act = request.args.get('act')

    cursor = mydb.cursor()
    
    cursor.execute("SELECT * FROM cf_criminal_details where id=%s",(vid,))
    value = cursor.fetchone()
    name=value[1]
    
    ff=open("user.txt","w")
    ff.write(name)
    ff.close()

    ff=open("user1.txt","w")
    ff.write(vid)
    ff.close()
    
  
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1="User."+vid+"."+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface="User."+vid+"."+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update cf_criminal_details set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('static/faces/f1.jpg', 'static/photo/'+vface1)

        
        ##########
        
        ##Training face
        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # function to get the images and label data
        

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))






        #################################################
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            #noice
            img = cv2.imread('static/process1/'+rs[2]) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2='ns_'+rs[2]
            cv2.imwrite("static/process1/"+fname2, dst)
            ######
            ##bin
            image = cv2.imread('static/process1/'+rs[2])
            original = image.copy()
            kmeans = kmeans_color_quantization(image, clusters=4)

            # Convert to grayscale, Gaussian blur, adaptive threshold
            gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
            
            # Draw largest enclosing circle onto a mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                break
            
            # Bitwise-and for result
            result = cv2.bitwise_and(original, original, mask=mask)
            result[mask==0] = (0,0,0)

            
            ###cv2.imshow('thresh', thresh)
            ###cv2.imshow('result', result)
            ###cv2.imshow('mask', mask)
            ###cv2.imshow('kmeans', kmeans)
            ###cv2.imshow('image', image)
            ###cv2.waitKey()

            cv2.imwrite("static/process1/bin_"+rs[2], thresh)
            

            ###RPN - Segment
            img = cv2.imread('static/process1/'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            ####
            img = cv2.imread('static/process2/fg_'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            '''
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            '''
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
        ###
        #cursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
        #cnt = cursor.fetchone()[0]
        #if cnt>10:
        return redirect(url_for('view_photo',vid=vid,act='success'))
        #else:
        #    return redirect(url_for('message',vid=vid))
    
    
    return render_template('add_photo.html',data=data, vid=vid)

def DeepCNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))


@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    uname=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()
    
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    
    value=[]
    
       
    mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
    value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        #vid=request.form['vid']
        
        #shutil.copy('static/img/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', data=data,result=value,vid=vid)



@app.route('/pro1',methods=['POST','GET'])
def pro1():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1

    
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    
    if act1<n:
        s1="1"
        
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
        
        
    else:
        s1="2"

    
    return render_template('pro1.html', value=value,vid=vid, act=act3,s1=s1)

@app.route('/pro2',methods=['POST','GET'])
def pro2():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None or act=='0':
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro2.html', value=value,vid=vid, act=act3,s1=s1)

@app.route('/pro3',methods=['POST','GET'])
def pro3():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro3.html', value=value,vid=vid, act=act3,s1=s1)

@app.route('/pro4',methods=['POST','GET'])
def pro4():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro4.html', value=value,vid=vid, act=act3,s1=s1)



@app.route('/pro5',methods=['POST','GET'])
def pro5():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro5.html', value=value,vid=vid, act=act3,s1=s1)

@app.route('/pro6',methods=['POST','GET'])
def pro6():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro6.html', value=value,vid=vid, act=act3,s1=s1)

@app.route('/pro7',methods=['POST','GET'])
def pro7():
    s1=""
    vid = request.args.get('vid')
    act = request.args.get('act')
    value=[]
    
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid, ))
    cnt = mycursor.fetchone()[0]

    if act is None:
        act=1
        
    act1=int(act)-1
    act2=int(act)+1
    act3=str(act2)
    
    n=10
    if act1<n:
        s1="1"
        mycursor.execute("SELECT * FROM vt_face where vid=%s limit %s,1",(vid, act1))
        value = mycursor.fetchone()
    else:
        s1="2"

    
    return render_template('pro7.html', value=value,vid=vid, act=act3,s1=s1)



@app.route('/view_criminal',methods=['POST','GET'])
def view_criminal():
    act=request.args.get("act")
    data2=[]
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    ff=open("facest.txt","r")
    face_st=ff.read()
    ff.close()


    
    mycursor.execute("SELECT * FROM cf_criminal_details where police_station=%s",(uname,))
    data2 = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cf_criminal_alert where cid=%s",(did,))
        mydb.commit()
        mycursor.execute("delete from cf_criminal_details where id=%s",(did,))
        mydb.commit()

        return redirect(url_for('view_criminal'))

    
    return render_template('view_criminal.html', data=data,data2=data2,act=act)

@app.route('/view_video',methods=['POST','GET'])
def view_video():
    act=request.args.get("act")
    data2=[]
    s1=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    ff=open("facest.txt","r")
    face_st=ff.read()
    ff.close()


    
    mycursor.execute("SELECT * FROM cf_video")
    vdata = mycursor.fetchall()
    if request.method=='POST':
        
        video=request.form['video']
        ff=open("video2.txt","w")
        ff.write(video)
        ff.close()
        s1="1"


    
    return render_template('view_video.html', data=data,data2=data2,act=act,vdata=vdata,s1=s1)

@app.route('/view_video1',methods=['POST','GET'])
def view_video1():
    act=request.args.get("act")
    data2=[]
    s1=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    ff=open("facest.txt","r")
    face_st=ff.read()
    ff.close()


    
    mycursor.execute("SELECT * FROM cf_video")
    vdata = mycursor.fetchall()
    if request.method=='POST':
        
        video=request.form['video']
        ff=open("video2.txt","w")
        ff.write(video)
        ff.close()
        s1="1"


    
    return render_template('view_video1.html', data=data,data2=data2,act=act,vdata=vdata,s1=s1)

@app.route('/view_report',methods=['POST','GET'])
def view_report():
    act=request.args.get("act")
    data2=[]
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_criminal_alert where station_id=%s order by id desc",(uname,))
    data = mycursor.fetchall()
    
    
    return render_template('view_report.html', data=data,act=act)

@app.route('/view_report2',methods=['POST','GET'])
def view_report2():
    act=request.args.get("act")
    data=[]
    s1=""
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_camera")
    cdata = mycursor.fetchall()
        

    if request.method=='POST':
        
        camera=request.form['camera']
        mycursor.execute("SELECT count(*) FROM cf_criminal_alert where camera=%s order by id desc",(camera,))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            s1="1"
            mycursor.execute("SELECT * FROM cf_criminal_alert where camera=%s order by id desc",(camera,))
            data = mycursor.fetchall()
        else:
            s1="2"

    
    return render_template('view_report2.html', data=data,act=act,cdata=cdata,s1=s1)

@app.route('/capture_video',methods=['POST','GET'])
def capture_video():
    act=request.args.get("act")
    uname=""
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()

    
    mycursor.execute("SELECT max(id)+1 FROM cf_video")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    filename="v"+str(maxid)+".avi"
    sql = "INSERT INTO cf_video(id,filename) VALUES (%s, %s)"
    val = (maxid,filename)
    mycursor.execute(sql, val)
    mydb.commit()

    ff=open("video.txt","w")
    ff.write(filename)
    ff.close()

    
    return render_template('capture_video.html')

@app.route('/captured',methods=['POST','GET'])
def captured():
    act=request.args.get("act")
    uname=""
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()

    if act=="del":
        did=request.args.get("did")

        mycursor.execute("SELECT * FROM cf_video where id=%s",(did,))
        d1 = mycursor.fetchone()
        os.remove("static/video/"+d1[1])
    
        mycursor.execute("delete from cf_video where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('captured'))

    mycursor.execute("SELECT * FROM cf_video")
    data = mycursor.fetchall()
    
    return render_template('captured.html', data=data,act=act)

@app.route('/add_crime2',methods=['POST','GET'])
def add_crime2():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
        
    act=request.args.get("act")
    cid=request.args.get("cid")

    mycursor = mydb.cursor()
    if request.method=='POST':
        
        details=request.form['details']
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")

        mycursor.execute("SELECT max(id)+1 FROM crime_info")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO crime_info(id,cid,details,uname,rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid,cid,details,uname,rdate)
        
        mycursor.execute(sql, val)
        mydb.commit()
        
        return redirect(url_for('add_crime2',cid=cid))

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from crime_info where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_crime2',cid=cid))

    mycursor.execute("SELECT * FROM crime_info where cid=%s",(cid,))
    data = mycursor.fetchall()
   
    return render_template('add_crime2.html',msg=msg,act=act,cid=cid,data=data)


@app.route('/process',methods=['POST','GET'])
def process():
    msg=""
    ss=""
    uname=""
    mobile=""
    mess=""
    act=""
    det=""
    st=""
   

    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    
    #print("uname="+uname)
    #shutil.copy('static/faces/f1.jpg', 'static/f1.jpg')

    ff3=open("img.txt","r")
    mcnt=ff3.read()
    ff3.close()

    ff11=open("sms.txt","r")
    sms=ff11.read()
    ff11.close()

    mycursor = mydb.cursor()

    try:
    
        mcnt1=int(mcnt)
        print(mcnt1)
        if mcnt1>=2:
            msg="Face Detected"
            cutoff=10
            act="1"
            mycursor.execute('SELECT * FROM vt_face')
            dt = mycursor.fetchall()
            for rr in dt:
                hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
                hash1 = imagehash.average_hash(Image.open("static/faces/f1.jpg"))
                cc1=hash0 - hash1
                print("cc="+str(cc1))
                if cc1<=cutoff:
                    vid=rr[1]
                    mycursor.execute('SELECT * FROM register where id=%s',(vid,))
                    rw = mycursor.fetchone()
                    name=rw[1]
                    msg="Name: "+rw[1]+" , Criminal Identified"
                    uu=rw[5]
                    mess="Criminal: "+name+", found in Cam01"
                    print(mess)
                    if uu=="admin":
                        mycursor.execute('SELECT * FROM police order by id desc limit 0,1')
                        ds = mycursor.fetchone()
                        mobile=str(ds[2])
                    else:
                        mycursor.execute('SELECT * FROM police where uname=%s',(uu,))
                        ds = mycursor.fetchone()
                        mobile=str(ds[2])
                    ##
                    print(mobile)
                    mycursor.execute("SELECT max(id)+1 FROM alert_info")
                    maxid = mycursor.fetchone()[0]
                    if maxid is None:
                        maxid=1

                    fimg="F"+str(maxid)+".jpg"
                    shutil.copy('static/faces/f1.jpg', 'static/detect/'+fimg)
                    sql = "INSERT INTO alert_info(id, cid, name, fimg) VALUES (%s, %s, %s, %s)"
                    val = (maxid,vid,name,fimg)
                    print(sql)
                    mycursor.execute(sql, val)
                    mydb.commit()

                    ff11=open("sms.txt","w")
                    ff11.write("2")
                    ff11.close()

                    if sms=="1":
                        st="1"
                    ##
            
                    mst=rr[3]
                    
                                     
                    break
                else:
                    msg="Not Identified"
    except:
        print("excep")
        
    #elif mcnt1>2:
    #    msg="Detected!"
    #else:
    #    msg="Face not Detected"
        
    return render_template('process.html',msg=msg,act=act,mcnt1=mcnt1,det=det,mess=mess,mobile=mobile,st=st)

@app.route('/verify_criminal',methods=['POST','GET'])
def verify_criminal():
    msg=""
    act=request.args.get("act")
    uname=""
    cid=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']


    mycursor = mydb.cursor()

    
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()


    
        

    return render_template('verify_criminal.html',msg=msg,data=data,cid=cid,act=act)

@app.route('/verifyy',methods=['POST','GET'])
def verifyy():
    msg=""
    uname=""
    cid=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()



        
    return render_template('verifyy.html',msg=msg)

@app.route('/verify',methods=['POST','GET'])
def verify():
    msg=""
    uname=""
    cid=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()

    ff=open("facest.txt","r")
    fdata=ff.read()
    ff.close()

    if fdata=="":
        s=1
    else:
        mycursor.execute("SELECT count(*) FROM cf_criminal_details where id=%s",(str(fdata),))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            mycursor.execute("SELECT * FROM cf_criminal_details where id=%s",(str(fdata),))
            dd=mycursor.fetchone()
            msg="yes"

            cid=str(dd[0])
        else:
            msg="no"

        
    return render_template('verify.html',msg=msg,data=data,cid=cid)



@app.route('/identify',methods=['POST','GET'])
def identify():
    uname=""
    msg=""
    cid = request.args.get('cid')
    if 'username' in session:
        uname = session['username']

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()
    


    mycursor.execute("SELECT * FROM cf_criminal_details where id=%s",(cid,))
    data2 = mycursor.fetchone()

        
    return render_template('identify.html',msg=msg,data=data,cid=cid,data2=data2)

##YOLOv8 - for Real Time Predictions
class YoloDetector:
    def __init__(self, weights_name='yolov8n_state_dict.pt', config_name='yolov8n.yaml', device='cuda:0', min_face=100, target_size=None, frontal=False):
           
            self._class_path = pathlib.Path(__file__).parent.absolute()#os.path.dirname(inspect.getfile(self.__class__))
            self.device = device
            self.target_size = target_size
            self.min_face = min_face
            self.frontal = frontal
            if self.frontal:
                print('Currently unavailable')
                # self.anti_profile = joblib.load(os.path.join(self._class_path, 'models/anti_profile/anti_profile_xgb_new.pkl'))
            self.detector = self.init_detector(weights_name,config_name)

    def init_detector(self,weights_name,config_name):
        print(self.device)
        model_path = os.path.join(self._class_path,'weights/',weights_name)
        print(model_path)
        config_path = os.path.join(self._class_path,'models/',config_name)
        state_dict = torch.load(model_path)
        detector = Model(cfg=config_path)
        detector.load_state_dict(state_dict)
        detector = detector.to(self.device).float().eval()
        for m in detector.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return detector
    
    def _preprocess(self,imgs):
        """
            Preprocessing image before passing through the network. Resize and conversion to torch tensor.
        """
        pp_imgs = []
        for img in imgs:
            h0, w0 = img.shape[:2]  # orig hw
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:  
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

            imgsz = check_img_size(max(img.shape[:2]), s=self.detector.stride.max())  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            pp_imgs.append(img)
        pp_imgs = np.array(pp_imgs)
        pp_imgs = pp_imgs.transpose(0, 3, 1, 2)
        pp_imgs = torch.from_numpy(pp_imgs).to(self.device)
        pp_imgs = pp_imgs.float()  # uint8 to fp16/32
        pp_imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        return pp_imgs
        
    def _postprocess(self, imgs, origimgs, pred, conf_thres, iou_thres):
        """
            Postprocessing of raw pytorch model output.
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        """
        bboxes = [[] for i in range(len(origimgs))]
        landmarks = [[] for i in range(len(origimgs))]
        
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        
        for i in range(len(origimgs)):
            img_shape = origimgs[i].shape
            h,w = img_shape[:2]
            gn = torch.tensor(img_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn_lks = torch.tensor(img_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            det = pred[i].cpu()
            scaled_bboxes = scale_coords(imgs[i].shape[1:], det[:, :4], img_shape).round()
            scaled_cords = scale_coords_landmarks(imgs[i].shape[1:], det[:, 5:15], img_shape).round()

            for j in range(det.size()[0]):
                box = (det[j, :4].view(1, 4) / gn).view(-1).tolist()
                box = list(map(int,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]))
                if box[3] - box[1] < self.min_face:
                    continue
                lm = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                lm = list(map(int,[i*w if j%2==0 else i*h for j,i in enumerate(lm)]))
                lm = [lm[i:i+2] for i in range(0,len(lm),2)]
                bboxes[i].append(box)
                landmarks[i].append(lm)
        return bboxes, landmarks

    def get_frontal_predict(self, box, points):
        '''
            Make a decision whether face is frontal by keypoints.
            Returns:
                True if face is frontal, False otherwise.
        '''
        cur_points = points.astype('int')
        x1, y1, x2, y2 = box[0:4]
        w = x2-x1
        h = y2-y1
        diag = sqrt(w**2+h**2)
        dist = scipy.spatial.distance.pdist(cur_points)/diag
        predict = self.anti_profile.predict(dist.reshape(1, -1))[0]
        if predict == 0:
            return True
        else:
            return False
    def align(self, img, points):
        '''
            Align faces, found on images.
            Params:
                img: Single image, used in predict method.
                points: list of keypoints, produced in predict method.
            Returns:
                crops: list of croped and aligned faces of shape (112,112,3).
        '''
        crops = [align_faces(img,landmark=np.array(i)) for i in points]
        return crops

    def predict(self, imgs, conf_thres = 0.3, iou_thres = 0.5):
        '''
            Get bbox coordinates and keypoints of faces on original image.
            Params:
                imgs: image or list of images to detect faces on
                conf_thres: confidence threshold for each prediction
                iou_thres: threshold for NMS (filtering of intersecting bboxes)
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        '''
        one_by_one = False
        # Pass input images through face detector
        if type(imgs) != list:
            images = [imgs]
        else:
            images = imgs
            one_by_one = False
            shapes = {arr.shape for arr in images}
            if len(shapes) != 1:
                one_by_one = True
                warnings.warn(f"Can't use batch predict due to different shapes of input images. Using one by one strategy.")
        origimgs = copy.deepcopy(images)
        
        
        if one_by_one:
            images = [self._preprocess([img]) for img in images]
            bboxes = [[] for i in range(len(origimgs))]
            points = [[] for i in range(len(origimgs))]
            for num, img in enumerate(images):
                with torch.inference_mode(): # change this with torch.no_grad() for pytorch <1.8 compatibility
                    single_pred = self.detector(img)[0]
                    print(single_pred.shape)
                bb, pt = self._postprocess(img, [origimgs[num]], single_pred, conf_thres, iou_thres)
                #print(bb)
                bboxes[num] = bb[0]
                points[num] = pt[0]
        else:
            images = self._preprocess(images)
            with torch.inference_mode(): # change this with torch.no_grad() for pytorch <1.8 compatibility
                pred = self.detector(images)[0]
            bboxes, points = self._postprocess(images, origimgs, pred, conf_thres, iou_thres)

        return bboxes, points
###Surveillance############
@app.route('/monitor',methods=['POST','GET'])
def monitor():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    ff3=open("camera.txt","r")
    camera=ff3.read()
    ff3.close()
    
    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
        
    mycursor = mydb.cursor()
   
                
    return render_template('monitor.html',msg=msg,camera=camera)

@app.route('/verify_image',methods=['POST','GET'])
def verify_image():
    msg=""
    act=request.args.get("act")
    uname=""
    mess=""
    mobile=""
    name=""
    sms=""
    data=[]
    data2=[]
    s1=""

    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(uname,))
    data = mycursor.fetchone()


    mycursor.execute("SELECT * FROM cf_video")
    vdata = mycursor.fetchall()
    if request.method=='POST':
        
        video=request.form['video']
        ff=open("video2.txt","w")
        ff.write(video)
        ff.close()
        
        ##
        path_main = 'static/test'
        for fname in os.listdir(path_main):
            if os.path.isfile(path_main+"/"+fname):
                os.remove(path_main+"/"+fname)
        ###
        if os.path.isfile("trainer1/trainer.yml"):
            os.remove("trainer1/trainer.yml")
        ###
        filename=""
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            fr1=randint(5,20)
            fr2=randint(20,70)
            vm1=str(fr1)
            vm2=str(fr2)
            fn="User."+vm1+"."+vm2+".jpg"
            
            file.save(os.path.join("static/test", fn))
        ##
        # Path for face image database
        path = 'static/test'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer1/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

        s1="1"

    
    return render_template('verify_image.html', msg=msg,data=data,data2=data2,act=act,vdata=vdata,s1=s1)


@app.route('/verify_image1',methods=['POST','GET'])
def verify_image1():
    msg=""
    act=request.args.get("act")
    uname=""
    mess=""
    mobile=""
    name=""
    sms=""
    data=[]
    data2=[]
    s1=""

    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()



    mycursor.execute("SELECT * FROM cf_video")
    vdata = mycursor.fetchall()
    if request.method=='POST':
        
        video=request.form['video']
        ff=open("video2.txt","w")
        ff.write(video)
        ff.close()
        
        ##
        path_main = 'static/test'
        for fname in os.listdir(path_main):
            if os.path.isfile(path_main+"/"+fname):
                os.remove(path_main+"/"+fname)
        ###
        if os.path.isfile("trainer1/trainer.yml"):
            os.remove("trainer1/trainer.yml")
        ###
        filename=""
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            fr1=randint(5,20)
            fr2=randint(20,70)
            vm1=str(fr1)
            vm2=str(fr2)
            fn="User."+vm1+"."+vm2+".jpg"
            
            file.save(os.path.join("static/test", fn))
        ##
        # Path for face image database
        path = 'static/test'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer1/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

        s1="1"

    
    return render_template('verify_image1.html', msg=msg,data=data,data2=data2,act=act,vdata=vdata,s1=s1)

@app.route('/showframe',methods=['POST','GET'])
def showframe():
    msg=""
    value=[]
    frdata=[]
    s1=""
    ff=open("static/frames2.txt","r")
    vv=ff.read()
    ff.close()

    ff=open("static/fstatus.txt","r")
    res=ff.read()
    ff.close()

    if vv=="":
        s=1
    else:
        s1="1"
        value=vv.split("|")

    return render_template('showframe.html',frdata=frdata,value=value,s1=s1,res=res)    

@app.route('/verify2',methods=['POST','GET'])
def verify2():
    msg=""
    uname=""
    mess=""
    mobile=""
    name=""
    sms=""
    cid=""
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()


    ff=open("facest.txt","r")
    fdata=ff.read()
    ff.close()

    ff=open("camera.txt","r")
    cam=ff.read()
    ff.close()

    ff=open("sms.txt","r")
    sm=ff.read()
    ff.close()

    sm1=int(sm)+1
    sm2=str(sm1)

    if fdata=="" or fdata=="no":
        s=1
    else:
        mycursor.execute("SELECT count(*) FROM cf_criminal_details where id=%s",(str(fdata),))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            mycursor.execute("SELECT * FROM cf_criminal_details where id=%s",(str(fdata),))
            dd=mycursor.fetchone()
            msg="yes"
            name=dd[1]
            cid=str(dd[0])
            #fimg=dd[19]
            

            ff=open("sms.txt","w")
            ff.write(sm2)
            ff.close()

            mycursor.execute("SELECT * FROM cf_camera where camera=%s",(cam,))
            dd2=mycursor.fetchone()
            sid=dd2[2]

            mycursor.execute("SELECT * FROM cf_policestation where station_id=%s",(sid,))
            dd1=mycursor.fetchone()
            area=dd1[4]
            city=dd1[5]
            mobile=dd1[2]
            #camera=dd1[9]

            if sm1<3:
                sms="1"

            mess=name+"- Criminal Detected"

            mycursor.execute("SELECT max(id)+1 FROM cf_criminal_alert")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            fimg="F"+str(maxid)+".jpg"
            shutil.copy('static/faces/f1.jpg', 'static/detect/'+fimg)
            sql = "INSERT INTO cf_criminal_alert(id, cid, name, face_image, camera, area, city, station_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,cid,name,fimg,cam,area,city,sid)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()
        else:
            msg="no"

        
    return render_template('verify2.html',msg=msg,cid=cid,mobile=mobile,mess=mess,sms=sms,name=name)







@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))
##########################################################
def gen5(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed4')
def video_feed5():
    return Response(gen5(VideoCamera5()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################
def gen4(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed4')
def video_feed4():
    return Response(gen4(VideoCamera4()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################
def gen3(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed3')
        

def video_feed3():
    return Response(gen3(VideoCamera3()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
##########################################################
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
###############################
def gen2(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed2')       
def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
