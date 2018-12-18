#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
import feature_extraction 

from tqdm import tqdm
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from skimage.feature import haar_like_feature
from skimage.feature import local_binary_pattern as lbp
from skimage.io import imread
from skimage.transform import pyramid_gaussian, integral_image
from sklearn.externals import joblib

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Train the Classifier


# In[ ]:


def train_classifier(args=object):
    fds = []
    labels = []
    
    print("==> Loading the positive features")
    for feat_path in tqdm(glob.glob(os.path.join(args.DIR_PATHS["POS_FEAT_PH"], "*.feat"))):
        fd = joblib.load(feat_path)
        fds.append(fd.reshape(-1))
        labels.append(1)

    print("==> Load the negative features")
    for feat_path in tqdm(glob.glob(os.path.join(args.DIR_PATHS["NEG_FEAT_PH"], "*.feat"))):
        fd = joblib.load(feat_path)
        fds.append(fd.reshape(-1))
        labels.append(0)

    if args.CLF_TYPE is "LIN_SVM":
        clf = LinearSVC()
        print("==> Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        joblib.dump(clf, args.MODEL_PH)
        print("==> Classifier saved to {}".format(args.MODEL_PH))
    elif args.CLF_TYPE is "MLP":
        clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(16, 32, 64), random_state=1)
        print("==> Training a Multi Layer Classifier")
        clf.fit(fds, labels)
        joblib.dump(clf, args.MODEL_PH)
        print("==> Classifier saved to {}".format(args.MODEL_PH))


# In[ ]:


# Perform Non-Maxima Suppression


# In[ ]:


def overlapping_area(detection_1, detection_2):
    """
        Function to calculate overlapping area'si
        `detection_1` and `detection_2` are 2 detections whose area
        of overlap needs to be found out.
        Each detection is list in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        The function returns a value between 0 and 1,
        which represents the area of overlap.
        0 is no overlap and 1 is complete overlap.
        Area calculated from ->
        http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    """
    # Calculate the x-y co-ordinates of the rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


# In[ ]:


def nms(detections, threshold=.5):
    """
        This function performs Non-Maxima Suppression.
        `detections` consists of a list of detections.
        Each detection is in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        If the area of overlap is greater than the `threshold`,
        the area with the lower confidence score is removed.
        The output is a list of detections.
    """
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
                        reverse=True)
    new_detections = [] # Unique detections will be appended to this list
    new_detections.append(detections[0]) # Append the first detection
    del detections[0] # Remove the detection from the original list
    """
        For each detection, calculate the overlapping area
        and if area of overlap is less than the threshold set
        for the detections in `new_detections`, append the 
        detection to `new_detections`.
        In either case, remove the detection from `detections` list.
    """
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections


# In[ ]:


# Test the classifier


# In[ ]:


def resize_by_short(img, short_len=256):
    print(img.size)
    (x, y) = img.size
    if x > y:
        y_s = short_len
        x_s = int(x * y_s / y)
        img = img.resize((x_s, y_s))
    else:
        x_s = short_len
        y_s = int(y * x_s / x)
        img = img.resize((x_s, y_s))
    return img


def sliding_window(image, window_size, step_size):
    """
        This function returns a patch of the input image `image` of size equal
        to `window_size`. The first image returned top-left co-ordinates (0, 0)
        and are increment in both x and y directions by the `step_size` supplied.
        So, the input parameters are -
        * `image` - Input Image
        * `window_size` - Size of Sliding Window
        * `step_size` - Incremented Size of Window

        The function returns a tuple -
        (x, y, im_window)
        where
        * x is the top-left x co-ordinate
        * y is the top-left y co-ordinate
        * im_window is the sliding window image
    """
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def test_classifier(args=object):
    for im_path in [os.path.join(args.DIR_PATHS["TEST_IMG_DIR_PH"], i) for i in os.listdir(args.DIR_PATHS["TEST_IMG_DIR_PH"]) if not i.startswith('.')]:
        # Read the Image
        im = Image.open(im_path).convert('L')
        im = np.array(resize_by_short(im))

        clf = joblib.load(args.MODEL_PH) # Load the classifier
        detections = [] # List to store the detections
        scale = 0 # The current scale of the image

        # Downscale the image and iterate
        for im_scaled in pyramid_gaussian(im, downscale=args.DOWNSCALE):
            cd = [] # This list contains detections at the current scale
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            if im_scaled.shape[0] < args.MIN_WDW_SIZE[1] or im_scaled.shape[1] < args.MIN_WDW_SIZE[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, args.MIN_WDW_SIZE, args.STEP_SIZE):
                if im_window.shape[0] != args.MIN_WDW_SIZE[1] or im_window.shape[1] != args.MIN_WDW_SIZE[0]:
                    continue
                # Calculate the HOG features
                fd = feature_extraction.process_image(im_window,args).reshape([1, -1])
                pred = clf.predict(fd)
                if pred == 1:
                    if args.IF_PRINT: print("==> Detection:: Location -> ({}, {})".format(x, y))
                    if args.CLF_TYPE is "LIN_SVM":
                        if args.IF_PRINT: print("==> Scale ->  {} Confidence Score {} \n".format(scale, clf.decision_function(fd)))
                        detections.append((x, y, clf.decision_function(fd),
                                           int(args.MIN_WDW_SIZE[0] * (args.DOWNSCALE ** scale)),
                                           int(args.MIN_WDW_SIZE[1] * (args.DOWNSCALE ** scale))))
                    elif args.CLF_TYPE is "MLP":
                        if args.IF_PRINT: print("==> Scale ->  {} Confidence Score {} \n".format(scale, clf.predict_proba(fd)[0][1]))#clf.decision_function(fd)))
                        detections.append((x, y, clf.predict_proba(fd)[0][1],
                                           int(args.MIN_WDW_SIZE[0] * (args.DOWNSCALE ** scale)),
                                           int(args.MIN_WDW_SIZE[1] * (args.DOWNSCALE ** scale))))
                    cd.append(detections[-1])

                # If visualize is set to true, display the working of the sliding window
                if args.VISUALIZE:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(30)

            # Move the the next scale
            scale += 1

        # Display the results before performing NMS
        clone = im.copy()

        # Draw the detections
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)

        detections_fin = nms(detections, args.THRESHOLD) # Perform Non Maxima Suppression

        # Display the results after performing NMS
        for (x_tl, y_tl, _, w, h) in detections_fin:
            # Draw the detections
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
            
        if args.VISUALIZE:
            cv2.imshow("Final Detections after applying NMS", clone)
            
        # print(os.path.split(im_path))
        print(os.path.join(args.DIR_PATHS['PRED_SAVE_PH'], os.path.split(im_path)[1]))
        cv2.imwrite(os.path.join(args.DIR_PATHS['PRED_SAVE_PH'], os.path.split(im_path)[1]), clone)


# In[ ]:



args = config.Config()
def train(args = args):
    feature_extraction.image_preprocess_size(args)
    feature_extraction.extract_features(args)
    train_classifier(args)
    test_classifier(args)
    


# In[ ]:


# Grid Search


# In[ ]:


#         self.MIN_WDW_SIZE = [64, 64]
#         self.STEP_SIZE = [12, 12]
#         self.ORIENTATIONS = 9
#         self.PIXELS_PER_CELL = [3, 3]
#         self.CELLS_PER_BLOCK = [3, 3]

import warnings
warnings.filterwarnings("ignore")

def grid_search():
    for step_size in range(8,36,4):
        for pixels_per_cell in range(3,10,1):
            for cells_per_block in range(3, 10, 1):
                args = config.Config()
                args.STEP_SIZE = [step_size, step_size]
                args.CELLS_PER_BLOCK = [cells_per_block, cells_per_block]
                args.PIXELS_PER_CELL = [pixels_per_cell, pixels_per_cell]
                args.PROJECT_ID = args.PROJECT_ID + "_SS_" + str(step_size) +                        "_CPB_" + str(cells_per_block) + "_PPC_" + str(pixels_per_cell)
                args.update_names()
                print(args.PROJECT_ID, args.DIR_PATHS)
                args.mk_new_dirs()
                feature_extraction.extract_features(args=args)
                train_classifier(args=args)
                test_classifier(args=args)
                if not args.KEEP_FEAT:
                    shutil.rmtree(args.DIR_PATHS['NEG_FEAT_PH'])
                    shutil.rmtree(args.DIR_PATHS['POS_FEAT_PH'])


# In[ ]:


#feature_extraction.image_preprocess_size(args)


# In[ ]:


#feature_extraction.extract_features(args)


# In[ ]:


#train_classifier(args)


# In[ ]:


#test_classifier(args)


# In[ ]:


grid_search()


# In[ ]:




