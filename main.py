import cv2
import numpy as np
import glob
import imutils
import os

print('\nFetching Images...')
folder_path = 'Intersection'
###EITHER PUT THE FULL PATH OF THE IMAGES FOLDER, OR OPEN THE WHOLE PROGRAM FOLDER IN VS CODE.
images_path = glob.glob(os.path.join(folder_path,'*.jpg')) 
images = []
for image in images_path:
    img = cv2.imread(image)
    images.append(img)

if images:
    print('>>Images Fetched!\n')  
     
    print('Stitching Images...\n(this may take a while)') 
    imageStitcher = cv2.Stitcher.create()
    error, stitched_image = imageStitcher.stitch(images=images)

    if not error:   
        ''' RAW STITCHED IMAGE '''
        cv2.imwrite('RawStitchedImage.png', stitched_image)
        print('>>Stitched Image Created!\n')
        
        
        ''' PROCESSING THE OUTPUT '''
        print('Processing into Panorama...')
        stitched_image = cv2.copyMakeBorder(stitched_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
        gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

        minRectangle = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)

        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(areaOI)

        stitched_image = stitched_image[y:y + h, x:x + w]
        
        ''' PROCESSED IMAGE '''
        cv2.imwrite("ProcessedStitchedImage.png", stitched_image)
        print('>>Panorama Created!\n')
        
    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")

else:
    print("\nERROR: No images found to stitch! Make sure you specified the right file path to the images.\n") 

