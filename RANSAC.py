import cv2 as cv
import numpy as np
    
#Path Configuration
SOURCE_PATH = 'source.jpg'
TARGET_PATH = 'template.jpg'
OUTPUT_PATH = 'result.jpg'

def myORB(source):
    
    target_image = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    height, width = target_image.shape

    template_image = cv.imread(TARGET_PATH)

    orb_target = cv.ORB_create(nfeatures=400000)
    kp2, des2 = orb_target.detectAndCompute(target_image, None)

    template_image = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)

    orb_template = cv.ORB_create(nfeatures=25000)
    kp1, des1 = orb_template.detectAndCompute(template_image,None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks = 32)
            
    flann = cv.FlannBasedMatcher(index_params, search_params)

    good = []
  
    matches = flann.knnMatch(des1,des2,k=2)
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                   
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0, maxIters = 100, confidence = 0.6)
        
        load_img = cv.imread('template.jpg')            
        h,w,c = load_img.shape
        pts = np.float32([ [1,1],[1,h-1],[w-1,h-1],[w-1,1] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)
        
        dst = np.squeeze(dst)
        
        y1 = dst[0,1] 
        y2 = dst[2,1]
        x1 = dst[0,0] 
        x2 = dst[2,0]
        
        for x in range(int(y1),int(y2)):
            for y in range(int(x1),int(x2)):
                work_image[x][y] = 0
                
        cv.rectangle(return_image, (x1,y1), (x2,y2), color=(255,0,0), thickness=5)

    return return_image


##################################################################################
#--------------------------------------DRIVER CODE-------------------------------#
##################################################################################

img = cv.imread(SOURCE_PATH)
work_image = img.copy()
return_image = img.copy()

result_img = myORB(work_image)

while(True):  
    test_img = result_img.copy()
    result_img = myORB(work_image)
   
    if(np.array_equal(result_img, test_img)):
        break;
    
cv.imwrite(OUTPUT_PATH, result_img)