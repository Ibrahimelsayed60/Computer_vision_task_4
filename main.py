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

        self.image_1 = cv2.rotate(cv2.imread("Threshold images\Lenna.png",0),cv2.ROTATE_90_CLOCKWISE)
        self.image_2 = cv2.rotate(cv2.imread("Threshold images\MRIbrain1.jpg",0),cv2.ROTATE_90_CLOCKWISE)

        self.ui.pushButton_1.clicked.connect(self.doing_otsu_global)
        self.ui.pushButton_2.clicked.connect(self.doing_otsu_local)
        self.ui.pushButton_3.clicked.connect(self.doing_original_image)
        


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
        self.ui.widget_1.clear()
        new_img = threshold.otsu_local_threshold(self.image_2,16)
        out = pg.ImageItem(new_img)
        self.ui.widget_1.addItem(out)
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()