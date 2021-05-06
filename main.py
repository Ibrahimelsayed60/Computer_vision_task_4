from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import sys
import math
import pyqtgraph as pg
import numpy as np

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


    def global_threshold_v_127(self,img):
        thresh = 100
        binary = img > thresh
        for i in range(0,len(binary),1):
            for j in range(0,len(binary),1):
                if binary[i][j] == True:
                    binary[i][j] = 256
                else:
                    binary[i][j]=0

        return binary

    
    def local_treshold(self,input_img):
        h, w = input_img.shape
        S = w/8
        s2 = S/2
        T = 15.0
        #integral img
        int_img = np.zeros_like(input_img, dtype=np.uint32)
        for col in range(w):
            for row in range(h):
                int_img[row,col] = input_img[0:row,0:col].sum()
        #output img
        out_img = np.zeros_like(input_img)    
        for col in range(w):
            for row in range(h):
                #SxS region
                y0 = int(max(row-s2, 0))
                y1 = int(min(row+s2, h-1))
                x0 = int(max(col-s2, 0))
                x1 = int(min(col+s2, w-1))
                count = (y1-y0)*(x1-x0)
                sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
                if input_img[row, col]*count < sum_*(100.-T)/100.:
                    out_img[row,col] = 0
                else:
                    out_img[row,col] = 255
        return out_img




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()