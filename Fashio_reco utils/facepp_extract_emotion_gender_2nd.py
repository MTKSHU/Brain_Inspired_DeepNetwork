#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:28:50 2018

@author: yo
"""
      
# -*- coding: utf-8 -*-
import urllib2
import urllib
import time
import json
import time
import os,shutil
#%%
path="/users/debopriyo/dataset/vggface/"
http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
key = ""
secret = ""
attributes = "age,gender,smiling,emotion,eyestatus,glass"
#emotion=''
log_file = open(os.path.join(path+"facepp_emotion_gender_extraction_2nd.log"),"a+")
test_list = open(path + "test_list.txt","r").readlines()

#meta_data_list_original=open(path+"test_list.txt","r").readlines()[:9228]
#meta_data_list_done=[]
#for folder in os.listdir(path+"test_img_metadata"):
#    meta_data_list_done+=[folder+"/"+x.split(".")[0]+".jpg\n" for x in os.listdir(path+"test_img_metadata/"+folder)]
#
#meta_data_list_rem_not_done=list(set(meta_data_list_original)-set(meta_data_list_done)) 

for img_path in test_list[102148:]:
    img_full_path = path + "test/" + img_path.rstrip("\n")

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(attributes)
    data.append('--%s' % boundary)
    
#    filepath = r"/users/debopriyo/sural/emc_v_1.1/data/face/new_static/img_3.jpeg"
    fr=open(img_full_path,'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename="i1.jpg"' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s--\r\n' % boundary)
    
    http_body='\r\n'.join(data)
    #buld http request
    req=urllib2.Request(http_url)
    #header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.add_data(http_body)
    try:
        #post data to server
        resp = urllib2.urlopen(req, timeout=5)
        #get response
        qrcont = json.loads(resp.read())
        if not os.path.exists(os.path.join(path + "test_img_metadata/" + img_path.split("/")[0])):
#            shutil.rmtree(os.path.join(path + "test_img_metadata/" + img_path.split("/")[0]))
            os.mkdir(os.path.join(path + "test_img_metadata/" + img_path.split("/")[0]))
        with open(path+"test_img_metadata/"+img_path.split(".jpg")[0]+".json","w") as f:
            json.dump(qrcont,f,sort_keys=True,indent=4,separators=(",",": "))
        print "Done: "+str(img_path.rstrip("\n"))
    #    print qrcont
        #emotion and gender
    #    if qrcont['faces']!=[]:
    #        emotion_happy = qrcont['faces'][0]['attributes']['emotion']['happiness']
    #        emotion_sad = qrcont['faces'][0]['attributes']['emotion']['sadness']
    #        if emotion_happy >= 95.0:
    #            emotion = 'H'
    #        elif emotion_sad >= 95.0:
    #            emotion = 'S'
    #        gender = str(qrcont['faces'][0]['attributes']['gender']['value'])[0]
    #    sql = "INSERT INTO `EMOJI_CAPTCHA` (`ori_img`,`img_with_noise`,`gender`,`emotion`) VALUES (%s,%s,%s,%s)"
    #    cursor.execute(sql, f,f.split('_')[0]+'with_hurl_noise_'+f.split('_')[1],gender,emotion)
    except Exception as e:
        print "Error : "+str(e)+" : "+img_path
        log_file.write(str(e) +","+str(test_list.index(img_path))+","+img_path)
    time.sleep(1.2)
log_file.close()


