from PyQt5.QtCore import pyqtSignal, QRect, Qt, QRegExp
from PyQt5.QtGui import QImage, QPixmap,QIntValidator
from PyQt5.QtWidgets import (QWidget, QPushButton, QDesktopWidget,QRadioButton,QButtonGroup,
                             QHBoxLayout, QVBoxLayout, QMainWindow, QLabel, QGridLayout, QProgressBar, QCheckBox,QLineEdit)
from image_database import ImageDB
import cvhelper
import cv2

class ImageDropLabel(QLabel):
    dropImgSignal = pyqtSignal(object)

    def __init__(self):
        super(ImageDropLabel, self).__init__()
        self.setAcceptDrops(True)
        self.initUI()
        pass

    def initUI(self):
        self.setStyleSheet(
            'border: 2px dashed black;\
            border-radius:10px;\
            background-color: lightgray;\
            min-width:300px;\
            min-height:300px;'
        )
        self.setText('Drag image here')
        self.setAlignment(Qt.AlignCenter)

        pass

    def dragEnterEvent(self, event):
        event.acceptProposedAction()
        pass

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].path()[-4:] == '.jpg':
                print(urls[0].path()[1:])
                cv_image = cvhelper.cv_imread(urls[0].path()[1:])
                smooth_image = cvhelper.cv_getSmoothImg(cv_image)
                contour = cvhelper.cv_drawContourImg(smooth_image,thickness=5)
                qimage = cvhelper.cv_toQimage(contour)
                qimage = qimage.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.dropImgSignal.emit(str(urls[0].path()[1:]))
                self.setPixmap(QPixmap(qimage))
        pass


class SortView(QWidget):
    def __init__(self):
        super(SortView, self).__init__()
        self.initUI()
        self.setVisible(False)

    def initUI(self):
        grid = QGridLayout()
        grid.setSpacing(5)

        for i in range(4):
            for j in range(5):
                simple_box = SimpleImageBox()
                grid.addWidget(simple_box, i, j)

        self.setLayout(grid)
        pass

    '''
    @SLOT
    Update Similarities
    '''
    def updateSimilarity(self, sorted_list):
        for i in range(20):
            self.layout().itemAt(i).widget().setName(sorted_list[i][0].split('\\')[-1].split('.')[0])
            self.layout().itemAt(i).widget().setSimilarity(sorted_list[i][1][0])
            self.layout().itemAt(i).widget().setImage(sorted_list[i][0],sorted_list[i][1][1],sorted_list[i][1][2])
            self.layout().itemAt(i).widget().setAngle(sorted_list[i][1][2])
            self.layout().itemAt(i).widget().setFlip(sorted_list[i][1][1])

        self.setVisible(True)
        pass

class DragImageBox(QWidget):
    def __init__(self):
        super(DragImageBox, self).__init__()
        self.image_path = ''
        self.imdb = ImageDB('轮廓图')
        self.imnum = len(self.imdb.img_paths)
        self.initUI()

        self.img_label.dropImgSignal.connect(self.updateByImgDrop)
        self.start_button.clicked.connect(self.calSim)
        self.button_group.buttonClicked[int].connect(self.setMethod)
        self.resize_checkbox.stateChanged.connect(self.setResize)
        self.sample_checkbox.stateChanged.connect(self.setSample)
        self.sample_edit.textChanged.connect(self.setSr)
        self.iter_edit.textChanged.connect(self.setIters)
        self.angle_edit.textChanged.connect(self.setAngleStep)
        self.resize_w.textChanged.connect(self.setW)
        self.resize_h.textChanged.connect(self.setH)
        self.smooth.stateChanged.connect(self.setSmooth)
        self.rotation_invariant.stateChanged.connect(self.setRotInv)
        self.show_interm.stateChanged.connect(self.setVisualize)
        self.imdb.step_signal.connect(self.updatePbar)
        self.imdb.done_signal.connect(self.endCal)

    def initUI(self):
        self.min_w = 128
        self.max_w = 2048
        self.min_h = 128
        self.max_h = 2048
        self.min_samples = 50
        self.max_samples = 500
        self.min_iters = 1
        self.max_iters = 10
        self.min_angle = 1
        self.max_angle = 360
        self.method = 'scd'
        self.img_label = ImageDropLabel()
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton()
        self.start_button.setText('开始')
        self.start_button.setEnabled(False)
        self.radio_hu = QRadioButton()
        self.radio_sc = QRadioButton()
        self.radio_haus = QRadioButton()
        QRegExp("[0-9]+$")
        self.sample_checkbox=QCheckBox()
        self.sample_checkbox.setText('Sample')
        self.sample_checkbox.setChecked(True)
        self.sample_edit =QLineEdit()
        self.sample_edit.setValidator(QIntValidator(self.min_samples, self.max_samples))
        self.sample_edit.setText(str(self.imdb.sample_rate))
        self.sample_edit.setPlaceholderText('%d> 采样点数 >%d' % (self.max_samples, self.min_samples))


        self.iter_label = QLabel('Iter')
        self.iter_edit = QLineEdit()
        self.iter_edit.setValidator(QIntValidator(self.min_iters, self.max_iters))
        self.iter_edit.setText(str(self.imdb.iterations))
        self.iter_edit.setPlaceholderText('%d> 迭代次数 >%d' % (self.max_iters, self.min_iters))

        self.angle_label = QLabel('Angle Step')
        self.angle_edit = QLineEdit()
        self.angle_edit.setValidator(QIntValidator(self.min_angle, self.max_angle))
        self.angle_edit.setText(str(self.imdb.rot_angle))
        self.angle_edit.setPlaceholderText('%d> 旋转角度 >%d' % (self.max_angle, self.min_angle))

        self.resize_checkbox=QCheckBox()
        self.resize_checkbox.setText('Resize')
        self.resize_checkbox.setChecked(self.imdb.do_resize)
        self.resize_w = QLineEdit()
        self.resize_w.setPlaceholderText('%d> 宽度 >%d'%(self.max_w, self.min_w))
        self.resize_h = QLineEdit()
        self.resize_h.setPlaceholderText('%d> 高度 >%d'%(self.max_h, self.min_h))
        self.resize_w.setValidator(QIntValidator(self.min_w, self.max_w))
        self.resize_h.setValidator(QIntValidator(self.min_h, self.max_h))
        self.resize_w.setText(str(self.imdb.resize_w))
        self.resize_h.setText(str(self.imdb.resize_h))
        self.smooth = QCheckBox()
        self.smooth.setText('Smooth')
        self.smooth.setChecked(False)

        self.rotation_invariant = QCheckBox()
        self.rotation_invariant.setText('Rotation Invariant')
        self.rotation_invariant.setChecked(False)

        self.show_interm = QCheckBox()
        self.show_interm.setText('可视化')
        self.show_interm.setChecked(False)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_hu, id=1)
        self.button_group.addButton(self.radio_sc, id=2)
        self.button_group.addButton(self.radio_haus, id=3)
        self.radio_sc.setChecked(True)
        self.progress_bar = ProgressBar(max_val=self.imnum)
        self.progress_label = QLabel()
        self.progress_label.setText('进度:')
        self.radio_hu.setText('Hu moments(Smaller is more similar)')
        self.radio_haus.setText('Hausdorff distance(Smaller is more similar)')
        self.radio_sc.setText('Shape context(Smaller is more similar)')


        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.img_label)
        vbox.addWidget(self.name_label)
        vbox.addWidget(self.radio_hu)
        vbox.addWidget(self.radio_sc)
        vbox.addWidget(self.radio_haus)
        vbox.addWidget(self.start_button)
        vhbox0 = QHBoxLayout()
        vhbox0.addWidget(self.sample_checkbox)
        vhbox0.addWidget(self.sample_edit)
        vbox.addLayout(vhbox0)

        vhbox1 = QHBoxLayout()
        vhbox1.addWidget(self.resize_checkbox)
        vhbox1.addWidget(self.resize_w)
        vhbox1.addWidget(self.resize_h)
        vbox.addLayout(vhbox1)
        vhbox2 = QHBoxLayout()
        # vhbox2.addWidget(self.iter_label)
        # vhbox2.addWidget(self.iter_edit)
        vhbox2.addWidget(self.smooth)
        vhbox2.addWidget(self.rotation_invariant)

        vhbox3 = QHBoxLayout()
        vhbox3.addWidget(self.iter_label)
        vhbox3.addWidget(self.iter_edit)

        vhbox4 = QHBoxLayout()
        vhbox4.addWidget(self.angle_label)
        vhbox4.addWidget(self.angle_edit)

        vbox.addLayout(vhbox2)
        vbox.addLayout(vhbox3)
        vbox.addLayout(vhbox4)
        progress_bar_hbox = QHBoxLayout()
        progress_bar_hbox.addWidget(self.progress_label)
        progress_bar_hbox.addWidget(self.progress_bar)
        vbox.addLayout(progress_bar_hbox)
        vbox.addStretch(1)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)
        self.setLayout(hbox)

    def updateByImgDrop(self, img_path):
        self.name_label.setText(img_path.split('/')[-1])
        self.image_path = img_path
        self.start_button.setEnabled(True)
        self.imdb.set_query_path(self.image_path)

        pass

    def calSim(self):
        self.sample_edit.setText(str(self.imdb.sample_rate))
        self.resize_h.setText(str(self.imdb.resize_h))
        self.resize_w.setText(str(self.imdb.resize_w))
        self.angle_edit.setText(str(self.imdb.rot_angle))
        self.iter_edit.setText(str(self.imdb.iterations))

        self.imdb.start()
        self.start_button.setEnabled(False)
        self.progress_bar.setStep(1)

    def endCal(self):
        print('done')
        self.progress_bar.setStep(1)
        self.start_button.setEnabled(True)

    def updatePbar(self,step):
        self.progress_bar.setStep(step)
        self.update()

    def setMethod(self, id):
        if id==1:
            self.method='hu'
        elif id==2:
            self.method='scd'
        elif id==3:
            self.method='hausdorff'

        self.imdb.set_method(self.method)
        pass

    def setResize(self, state):
        if state == Qt.Unchecked:
            self.imdb.do_resize = False
        else:
            self.imdb.do_resize = True
        pass

    def setSample(self,state):
        if state == Qt.Unchecked:
            self.imdb.do_sample = False
        else:
            self.imdb.do_sample = True
        pass

    def setSmooth(self,state):
        if state == Qt.Unchecked:
            self.imdb.do_smooth = False
        else:
            self.imdb.do_smooth = True
        pass

    def setRotInv(self, state):
        if state == Qt.Unchecked:
            self.imdb.rotation_invariant = False
        else:
            self.imdb.rotation_invariant = True
        pass

    def setVisualize(self, state):
        if state == Qt.Unchecked:
            self.imdb.visualize = False
        else:
            self.imdb.visualize = True
        pass

    def setIters(self, txt):
        if len(txt)==0 or int(txt) < self.min_iters:
            self.imdb.iterations = self.min_iters
        elif int(txt) > self.max_iters:
            self.imdb.iterations = self.max_iters
        else:
            self.imdb.iterations = int(txt)

    def setAngleStep(self, txt):
        if len(txt)==0 or int(txt) < self.min_angle:
            # self.resize_h.setText(str(self.min_h))
            self.imdb.rot_angle = self.min_angle
        elif int(txt) > self.max_angle:
            # self.resize_h.setText(str(self.max_h))
            self.imdb.rot_angle = self.max_angle
        else:
            self.imdb.rot_angle = int(txt)

    def setSr(self, txt):
        if len(txt)==0 or int(txt) < self.min_samples:
            self.imdb.sample_rate = self.min_samples
        elif int(txt) > self.max_samples:
            self.imdb.sample_rate = self.max_samples
        else:
            self.imdb.sample_rate = int(txt)

    def setW(self,txt):
        if len(txt)==0 or int(txt) < self.min_w:
            # self.resize_w.setText(str(self.min_w))
            self.imdb.resize_w = self.min_w
        elif int(txt) > self.max_w:
            # self.resize_w.setText(str(self.max_w))
            self.imdb.resize_w = self.max_w
        else:
            self.imdb.resize_w = int(txt)

    def setH(self, txt):
        if len(txt)==0 or int(txt) < self.min_h:
            # self.resize_h.setText(str(self.min_h))
            self.imdb.resize_h = self.min_h
        elif int(txt) > self.max_h:
            # self.resize_h.setText(str(self.max_h))
            self.imdb.resize_h = self.max_h
        else:
            self.imdb.resize_h = int(txt)



class SimpleImageBox(QWidget):
    def __init__(self):
        super(SimpleImageBox, self).__init__()
        self.initUI()


    def initUI(self):
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.name_label = QLabel()
        self.similarity_label = QLabel()
        self.flip_label = QLabel()
        self.angle_label = QLabel()
        vbox = QVBoxLayout()
        self.img_label.setFixedSize(150, 150)
        self.img_label.setStyleSheet('border: 2px dashed black; border-radius:10px; ')
        self.name_label.setFixedHeight(15)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.similarity_label.setFixedHeight(15)
        self.similarity_label.setAlignment(Qt.AlignCenter)
        self.similarity_label.setStyleSheet('color:green;')
        self.flip_label.setFixedHeight(15)
        self.flip_label.setAlignment(Qt.AlignCenter)
        self.angle_label.setFixedHeight(15)
        self.angle_label.setAlignment(Qt.AlignCenter)

        hbox = QHBoxLayout()
        hbox.addWidget(self.flip_label)
        hbox.addWidget(self.angle_label)


        vbox.addWidget(self.img_label)
        vbox.addWidget(self.name_label)
        vbox.addWidget(self.similarity_label)
        vbox.addLayout(hbox)

        hhbox = QHBoxLayout()
        hhbox.addStretch(1)
        hhbox.addLayout(vbox)
        hhbox.addStretch(1)
        self.setLayout(hhbox)
        pass

    def setSimilarity(self, sim):
        self.similarity_label.setText('%.6f'%sim)

    def setName(self, name):
        self.name_label.setText(str(name))

    def setImage(self, path, flip, rot_angle):
        cv_img = cvhelper.cv_imread(path)
        # if flip:
        #     cv_img = cv2.flip(cv_img, 1)
        # cv_img = cvhelper.cv_rotate(cv_img, rot_angle)
        qimg = cvhelper.cv_toQimage(cv_img)
        self.img_label.setPixmap(
            QPixmap(qimg.scaled(self.img_label.size(),Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))

    def setFlip(self,flip):
        if flip:
            self.flip_label.setText('翻转')
            self.flip_label.setStyleSheet("color:blue;")
        else:
            self.flip_label.setText('正常')
            self.flip_label.setStyleSheet("")

    def setAngle(self, angle):
        if angle > 0:
            self.angle_label.setText('旋转角度:%d' % angle)
            self.angle_label.setStyleSheet("color:blue;")
        else:
            self.angle_label.setText('无旋转')
            self.angle_label.setStyleSheet("")


class ProgressBar(QWidget):
    def __init__(self, max_val):
        super(ProgressBar,self).__init__()
        self.max_val = max_val
        self.initUI()


    def initUI(self):
        hbox = QHBoxLayout()
        self.pbar = QProgressBar(self)


        self.pbar.setMaximum(self.max_val)
        self.pbar.setMinimum(1)
        # self.setGeometry(300, 300, 280, 170)
        hbox.addWidget(self.pbar)
        self.setWindowTitle('进度条')
        self.setLayout(hbox)

    def setStep(self, step):
        self.pbar.setValue(step)


class CentralView(QWidget):
    def __init__(self):
        super(CentralView, self).__init__()
        self.initUI()


    def initUI(self):
        self.imageDropView = DragImageBox()
        self.sortView = SortView()


        self.imageDropView.imdb.done_signal.connect(self.sortView.updateSimilarity)
        vbox = QVBoxLayout()
        vbox.setSpacing(5)
        vbox.addWidget(self.imageDropView)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addWidget(self.sortView)
        hbox.setStretch(0, 3)
        hbox.setStretch(1, 9)
        self.setLayout(hbox)
        self.setWindowTitle('图片相似度排序demo')
        pass

    def updateBySimilarity(self):
        self.sortView.setVisible(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
        pass

    '''
    do some UI initializations here
    '''

    def initUI(self):
        self.centralView = CentralView()
        self.resize()
        self.setCentralWidget(self.centralView)
        self.show()

    '''
    resize widget to a ratio of the desktop and put it at the center
    '''

    def resize(self, ratio=0.8):
        desktop_rect = QDesktopWidget().availableGeometry()
        cp = desktop_rect.center()
        new_width = desktop_rect.width() * ratio
        new_height = desktop_rect.height() * ratio
        new_left = cp.x() - new_width // 2
        new_top = cp.y() - new_height // 2
        new_rect = QRect(new_left, new_top, new_width, new_height)
        self.setGeometry(new_rect)

    pass
