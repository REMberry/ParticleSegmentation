from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMessageBox
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
import pyqtgraph as pg
import numpy as np
import cv2

uiclass, baseclass = pg.Qt.loadUiType("mainwindow.ui")

debug_num = 5;

class MainWindow(uiclass, baseclass):
    def __init__(self):
        super().__init__()
        
        
        self.setupUi(self)
        
        self.pb_loadImage.clicked.connect(self.loadImage)
        self.pb_segment.clicked.connect(self.segmentImageClick)
        self.pb_hist.clicked.connect(self.showHistClick)

        pg.setConfigOptions(imageAxisOrder='row-major')
        demoImg = pg.gaussianFilter(np.random.normal(
            size=(200, 200,3)), (5,5,3)) * 1000 + 100
        demoImg = demoImg.astype(np.uint8)
        
        self.imageRaw  = demoImg
        self.image = pg.ImageItem(demoImg)
        self.image.setZValue(-0)
        
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image)
        self.hist.setFixedWidth(100)
        self.graphWidget.addItem(self.hist)
        
        self.plot1 = pg.PlotItem(title="")
        self.graphWidget.addItem(self.plot1)
        self.plot1.addItem(self.image)
        self.plot1.autoRange()
        self.plot1.setAspectLocked()
        self.plot1.invertY()
        
        #self.plot1 = self.graphWidget.addPlot(title="")
        #self.plot1.addItem(self.image)
        #self.plot1.autoRange()
        #self.plot1.setAspectLocked()
        #self.plot1.invertY()
        
        
        #self.plot1 = self.graphWidget.addItem(self.image)
        #self.plot1.addItem(self.image)
        #self.plot1.autoRange()
        #self.plot1.setAspectLocked()
        #self.plot1.invertY()
        #self.graphWidget.addItem(self.hist)

        # ROI - start with one
        #self.rois = []  # creaate array for containing the roi references
        #self.lW_rois.clear()  # clear roi's gui list
        #self.addRoi()  # add one roi

        # add line roi
        #self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
        #self.plot1.addItem(self.lineROI)
        #self.lineROI.sigRegionChangeFinished.connect(self.updatePlot2)

        #self.graphWidget.nextRow()
        #self.plot2 = self.graphWidget.addPlot(colspan = 3)
        #self.plot2.setMaximumHeight(120)
        #self.plot2.enableAutoRange()
        #self.plot1.enableAutoRange()

        #self.cB_autoScale.setChecked(True)
        #self.show()
        
        self.show()
        #self.image = pg.ImageItem()
        #self.graphWidget.addItem(self.image)
    def showHistClick(self):
    
        win = pg.GraphicsLayoutWidget(show=True)
        win.resize(800,480)
        win.setWindowTitle('pyqtgraph example: Histogram')
        plt1 = win.addPlot()
        plt2 = win.addPlot()
        
        ## make interesting distribution of values
        vals = np.hstack([np.random.normal(size=500), np.random.normal(size=260, loc=4)])
        
        ## compute standard histogram
        y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))
        
        ## Using stepMode="center" causes the plot to draw two lines for each sample.
        ## notice that len(x) == len(y)+1
        plt1.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150))
        
        ## Now draw all points as a nicely-spaced scatter plot
        psy = pg.pseudoScatter(vals, spacing=0.15)
        plt2.plot(vals, psy, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))
        
        # draw histogram using BarGraphItem
        win.nextRow()
        plt3 = win.addPlot()
        bgi = pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, pen='w', brush=(0,0,255,150))
        plt3.addItem(bgi)
        pg.exec()
    
    def segmentImageClick(self):
        
        self.startSegmentation()
    
    def startSegmentation(self):
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=10,  # Requires open-cv to run post-processing
        )
        
        print('Start Segmentation')
        self.masks = mask_generator.generate(self.imageRaw)
        print('Finish Segmentation')
        
        areas = [d['area'] for d in self.masks]
        print(areas)
 
        self.show_anns(self.masks )
        
        
        
    def list_attributes_and_values(self, obj):
        for attr_name in dir(obj):
            # Filtering out private and special attributes/methods
            if not attr_name.startswith("__"):
                try:
                    attr_value = getattr(obj, attr_name)
                    print(f"{attr_name} : {attr_value}")
                except Exception as e:
                    print(f"Couldn't retrieve value for {attr_name}. Reason: {e}")    
        
    
    def loadImage(self):
        print("debug")    
        
        options = QFileDialog.Options()
        
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image from File",
            "",
            """All (*.tiff *.png *.bmp *.jpeg *.tif);;
            TIFF File (*.tiff);;PNG File ( *.png);;
            TIF File (*.tif);;
            Bitmap (*.bmp);;Text File (*.txt);;
            CSV Table (*.csv);;JSON File (*.json);;
            Compressed Text File (*.gz)"""
            , options=options)
        
        if fileName:
            file = Path(fileName)
            
            if file.suffix.upper() in [".TIFF", ".PNG", ".BMP",".TIF"]:
                print('load image')
                
                image = cv2.imread(fileName)
                
                self.plot1.clear();
                
                self.imageRaw = image;
                self.image.setImage(image)
                self.plot1.addItem(self.image)
                self.plot1.autoRange()
                
    def show_anns(self,anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        #ax = plt.gca()
        #ax.set_autoscale_on(False)
    
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        #ax.imshow(img)    
        # Create an ImageItem from the mask image
        self.mask_layer = pg.ImageItem(img)
        self.mask_layer.setZValue(1)  # Ensure it's above the main image
    
        # Add mask layer to the plot
        self.plot1.addItem(self.mask_layer)
    

if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()

window = MainWindow()    
sys.exit(app.exec())

