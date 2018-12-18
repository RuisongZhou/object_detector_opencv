#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os


# In[8]:


class Config(object):
    def __init__(self, project_id=None):
        
        # General Config
        self.DES_TYPE = "HOG"
        self.CLF_TYPE = "LIN_SVM"
        if project_id:
            self.PROJECT_ID = project_id
        else:
            self.PROJECT_ID = "New_Vedio_New_Neg" + self.DES_TYPE + '_' + self.CLF_TYPE
        self.THRESHOLD = 0.3
        self.DOWNSCALE = 1.25
        
        # Pathes
        self.update_names()
        
        # HOG Features
        self.MIN_WDW_SIZE = [64, 64]
        self.STEP_SIZE = [12, 12]
        self.ORIENTATIONS = 9
        self.PIXELS_PER_CELL = [3, 3]
        self.CELLS_PER_BLOCK = [3, 3]
        self.VISUALIZE = False
        self.NORMALIZE = True
        self.IF_PRINT  = False
        self.KEEP_FEAT = False
        
        # LBP Features
        self.LBP_RADIUS = 3
        self.LBP_POINTS = 8 * self.LBP_RADIUS
        
        self.mk_new_dirs()
        
    def mk_new_dirs(self):
        for ph in self.DIR_PATHS.values():
            if not os.path.exists(ph):
                os.makedirs(ph)
                print("==> Directory Tree",ph,"created")
                
    def update_names(self):
        # Pathes
        self.DIR_PATHS = {
            "POS_FEAT_PH"    : os.path.join("./source/features", self.PROJECT_ID,"pos"),
            "NEG_FEAT_PH"    : os.path.join("./source/features", self.PROJECT_ID,"neg"),
            "MODEL_DIR_PH"   : os.path.join("./source/models", self.PROJECT_ID),
            "PRED_SAVE_PH"   : os.path.join("./source/predictions", self.PROJECT_ID),
            "POS_IMG_PH"     : "./source/images/pos",
            "NEG_IMG_PH"     : "./source/images/neg",
            "TEST_IMG_DIR_PH": "./source/test_images"}
        self.MODEL_PH = os.path.join(self.DIR_PATHS["MODEL_DIR_PH"], "svm.model")
        self.TEST_IMG_PH = os.path.join(self.DIR_PATHS["TEST_IMG_DIR_PH"], "test.jpg")
               



