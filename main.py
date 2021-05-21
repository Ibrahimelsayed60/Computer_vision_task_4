from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import sys
import math
import pyqtgraph as pg
import numpy as np
import threshold
import cv2
import segmentation
import optimal
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.widget_2.getPlotItem().hideAxis('bottom')
        self.ui.widget_2.getPlotItem().hideAxis('left')
        self.ui.widget_1.getPlotItem().hideAxis('bottom')
        self.ui.widget_1.getPlotItem().hideAxis('left')
        self.ui.widget_3.getPlotItem().hideAxis('bottom')
        self.ui.widget_3.getPlotItem().hideAxis('left')
        self.ui.widget.getPlotItem().hideAxis('bottom')
        self.ui.widget.getPlotItem().hideAxis('left')
        self.ui.widget_4.getPlotItem().hideAxis('bottom')
        self.ui.widget_4.getPlotItem().hideAxis('left')
        self.ui.widget_5.getPlotItem().hideAxis('bottom')
        self.ui.widget_5.getPlotItem().hideAxis('left')
        self.ui.widget_6.getPlotItem().hideAxis('bottom')
        self.ui.widget_6.getPlotItem().hideAxis('left')
        self.ui.widget_7.getPlotItem().hideAxis('bottom')
        self.ui.widget_7.getPlotItem().hideAxis('left')

        self.image_1 = cv2.rotate(cv2.imread("Threshold images\Lenna.png",0),cv2.ROTATE_90_CLOCKWISE)
        self.image_2 = cv2.rotate(cv2.imread("Threshold images\MRIbrain1.jpg",0),cv2.ROTATE_90_CLOCKWISE)
        image_3 = cv2.rotate(cv2.imread("Segmentation Images/sea.jpeg"),cv2.ROTATE_90_CLOCKWISE)
        self.image_3 = cv2.cvtColor(image_3, cv2.COLOR_RGB2Luv)
        out = pg.ImageItem(self.image_3)
        self.ui.widget.addItem(out)

        self.opimg = cv2.imread("Threshold images\Beads.jpg")
        self.ui.pushButton_1.clicked.connect(self.doing_otsu_global)
        self.ui.pushButton_3.clicked.connect(self.doing_original_image)
        self.ui.comboBox.currentIndexChanged[int].connect(self.segmentation)
        self.ui.comboBox_2.currentIndexChanged[int].connect(self.doing_otsu_local)
        self.ui.comboBox_3.currentIndexChanged[int].connect(self.optimal)


    def doing_global_threshold(self):
        new_img = threshold.global_threshold(self.image_1, 127)
        out = pg.ImageItem(new_img)
        #self.ui.widget_2.addItem(out)

    
    def doing_local_treshold(self):
        new_img = threshold.local_threshold(self.image_1)
        out = pg.ImageItem(new_img)
        #self.ui.widget_1.addItem(out)

    def doing_original_image(self):
        out = pg.ImageItem(self.image_2)
        self.ui.widget_2.addItem(out)


    def doing_otsu_global(self):
        self.ui.widget_3.clear()
        new_img = threshold.otsu_global_threshold(self.image_2)
        out = pg.ImageItem(new_img)
        self.ui.widget_3.addItem(out)

    def doing_otsu_local(self):
        if self.ui.comboBox_2.currentIndex() == 1:
            self.ui.widget_1.clear()
            new_img = threshold.otsu_local_threshold(self.image_2,16)
        elif self.ui.comboBox_2.currentIndex() == 2:
            self.ui.widget_1.clear()
            new_img = threshold.otsu_local_threshold(self.image_2,32)
        elif self.ui.comboBox_2.currentIndex() == 3:
            self.ui.widget_1.clear()
            new_img = threshold.otsu_local_threshold(self.image_2,64)
        elif self.ui.comboBox_2.currentIndex() == 4:
            self.ui.widget_1.clear()
            new_img = threshold.otsu_local_threshold(self.image_2,128)
        out = pg.ImageItem(new_img)
        self.ui.widget_1.addItem(out)
        
    def segmentation(self):
        if self.ui.comboBox.currentIndex() == 0:
            self.ui.widget_4.clear()
        if self.ui.comboBox.currentIndex() == 1:
            pixel_values = self.image_3.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            k = segmentation.KMeans(K=3, max_iters=100) 
            y_pred = k.predict(pixel_values)
            centers = np.uint8(k.cent())
            y_pred = y_pred.astype(int)
            labels = y_pred.flatten()
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(self.image_3.shape)
            out = pg.ImageItem(segmented_image)
            self.ui.widget_4.addItem(out)
        if self.ui.comboBox.currentIndex() == 2:
            meanshift = segmentation.meanshift(self.image_3)
            out = meanshift.performMeanShift(self.image_3)
            out = pg.ImageItem(out)
            self.ui.widget_4.addItem(out)

    def optimal(self):
        if self.ui.comboBox_3.currentIndex() == 0:
            self.ui.widget_5.clear()
            self.ui.widget_6.clear()
            self.ui.widget_7.clear()
        elif self.ui.comboBox_3.currentIndex() == 1:
            x = optimal.showop(self.opimg)
            out = pg.ImageItem(x)
            self.ui.widget_5.addItem(out)
        elif self.ui.comboBox_3.currentIndex() == 2:
            x = optimal.showloc(self.opimg)
            out = pg.ImageItem(x)
            self.ui.widget_6.addItem(out)

        elif self.ui.comboBox_3.currentIndex() == 3:
            x = optimal.showglob(self.opimg)
            out = pg.ImageItem(x)
            self.ui.widget_7.addItem(out)



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()