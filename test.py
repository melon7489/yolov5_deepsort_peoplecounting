# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(1212, 779)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setAcceptDrops(False)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockNestingEnabled(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_src2 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_src2.sizePolicy().hasHeightForWidth())
        self.label_src2.setSizePolicy(sizePolicy)
        self.label_src2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_src2.setObjectName("label_src2")
        self.gridLayout.addWidget(self.label_src2, 2, 0, 1, 1)
        self.label_det = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_det.sizePolicy().hasHeightForWidth())
        self.label_det.setSizePolicy(sizePolicy)
        self.label_det.setAlignment(QtCore.Qt.AlignCenter)
        self.label_det.setObjectName("label_det")
        self.gridLayout.addWidget(self.label_det, 0, 2, 1, 1)
        self.label_src = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_src.sizePolicy().hasHeightForWidth())
        self.label_src.setSizePolicy(sizePolicy)
        self.label_src.setAlignment(QtCore.Qt.AlignCenter)
        self.label_src.setObjectName("label_src")
        self.gridLayout.addWidget(self.label_src, 0, 0, 1, 1)
        self.label_det2 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_det2.sizePolicy().hasHeightForWidth())
        self.label_det2.setSizePolicy(sizePolicy)
        self.label_det2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_det2.setObjectName("label_det2")
        self.gridLayout.addWidget(self.label_det2, 2, 2, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.groupBox)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 0, 1, 3, 1)
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 0, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.groupBox)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 1, 2, 1, 1)
        self.line_13 = QtWidgets.QFrame(self.groupBox)
        self.line_13.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.gridLayout.addWidget(self.line_13, 0, 3, 3, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.openWebcam_front = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openWebcam_front.sizePolicy().hasHeightForWidth())
        self.openWebcam_front.setSizePolicy(sizePolicy)
        self.openWebcam_front.setMaximumSize(QtCore.QSize(9999, 50))
        self.openWebcam_front.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.openWebcam_front.setObjectName("openWebcam_front")
        self.verticalLayout_2.addWidget(self.openWebcam_front)
        self.open = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.open.sizePolicy().hasHeightForWidth())
        self.open.setSizePolicy(sizePolicy)
        self.open.setMaximumSize(QtCore.QSize(9999, 50))
        self.open.setObjectName("open")
        self.verticalLayout_2.addWidget(self.open)
        self.start = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start.sizePolicy().hasHeightForWidth())
        self.start.setSizePolicy(sizePolicy)
        self.start.setMaximumSize(QtCore.QSize(9999, 50))
        self.start.setObjectName("start")
        self.verticalLayout_2.addWidget(self.start)
        self.stop = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stop.sizePolicy().hasHeightForWidth())
        self.stop.setSizePolicy(sizePolicy)
        self.stop.setMaximumSize(QtCore.QSize(9999, 50))
        self.stop.setObjectName("stop")
        self.verticalLayout_2.addWidget(self.stop)
        self.line_3 = QtWidgets.QFrame(self.groupBox)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_2.addWidget(self.line_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.lcdNumber_out = QtWidgets.QLCDNumber(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber_out.sizePolicy().hasHeightForWidth())
        self.lcdNumber_out.setSizePolicy(sizePolicy)
        self.lcdNumber_out.setMinimumSize(QtCore.QSize(150, 0))
        self.lcdNumber_out.setMaximumSize(QtCore.QSize(400, 75))
        self.lcdNumber_out.setObjectName("lcdNumber_out")
        self.horizontalLayout.addWidget(self.lcdNumber_out)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.lcdNumber_in = QtWidgets.QLCDNumber(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber_in.sizePolicy().hasHeightForWidth())
        self.lcdNumber_in.setSizePolicy(sizePolicy)
        self.lcdNumber_in.setMinimumSize(QtCore.QSize(150, 0))
        self.lcdNumber_in.setMaximumSize(QtCore.QSize(400, 75))
        self.lcdNumber_in.setObjectName("lcdNumber_in")
        self.horizontalLayout_2.addWidget(self.lcdNumber_in)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.lcdNumber_online = QtWidgets.QLCDNumber(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber_online.sizePolicy().hasHeightForWidth())
        self.lcdNumber_online.setSizePolicy(sizePolicy)
        self.lcdNumber_online.setMinimumSize(QtCore.QSize(150, 0))
        self.lcdNumber_online.setMaximumSize(QtCore.QSize(400, 75))
        self.lcdNumber_online.setObjectName("lcdNumber_online")
        self.horizontalLayout_4.addWidget(self.lcdNumber_online)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.clearNum = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clearNum.sizePolicy().hasHeightForWidth())
        self.clearNum.setSizePolicy(sizePolicy)
        self.clearNum.setMaximumSize(QtCore.QSize(9999, 50))
        self.clearNum.setObjectName("clearNum")
        self.verticalLayout_2.addWidget(self.clearNum)
        self.close = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close.sizePolicy().hasHeightForWidth())
        self.close.setSizePolicy(sizePolicy)
        self.close.setMaximumSize(QtCore.QSize(9999, 50))
        self.close.setObjectName("close")
        self.verticalLayout_2.addWidget(self.close)
        self.line_6 = QtWidgets.QFrame(self.groupBox)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_2.addWidget(self.line_6)
        self.openWebcam_back = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openWebcam_back.sizePolicy().hasHeightForWidth())
        self.openWebcam_back.setSizePolicy(sizePolicy)
        self.openWebcam_back.setMaximumSize(QtCore.QSize(9999, 50))
        self.openWebcam_back.setObjectName("openWebcam_back")
        self.verticalLayout_2.addWidget(self.openWebcam_back)
        self.open2 = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.open2.sizePolicy().hasHeightForWidth())
        self.open2.setSizePolicy(sizePolicy)
        self.open2.setMaximumSize(QtCore.QSize(9999, 50))
        self.open2.setObjectName("open2")
        self.verticalLayout_2.addWidget(self.open2)
        self.start2 = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start2.sizePolicy().hasHeightForWidth())
        self.start2.setSizePolicy(sizePolicy)
        self.start2.setMaximumSize(QtCore.QSize(9999, 50))
        self.start2.setObjectName("start2")
        self.verticalLayout_2.addWidget(self.start2)
        self.stop2 = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stop2.sizePolicy().hasHeightForWidth())
        self.stop2.setSizePolicy(sizePolicy)
        self.stop2.setMaximumSize(QtCore.QSize(9999, 50))
        self.stop2.setObjectName("stop2")
        self.verticalLayout_2.addWidget(self.stop2)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 4, 3, 1)
        self.verticalLayout_4.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.action2 = QtWidgets.QAction(MainWindow)
        self.action2.setObjectName("action2")
        self.action3 = QtWidgets.QAction(MainWindow)
        self.action3.setObjectName("action3")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "test"))
        self.label.setText(_translate("MainWindow", "公交车场景人流量统计系统"))
        self.label_src2.setText(_translate("MainWindow", "源视频2"))
        self.label_det.setText(_translate("MainWindow", "检测结果1"))
        self.label_src.setText(_translate("MainWindow", "源视频1"))
        self.label_det2.setText(_translate("MainWindow", "检测结果2"))
        self.openWebcam_front.setText(_translate("MainWindow", "打开前门摄像头"))
        self.open.setText(_translate("MainWindow", "打开文件"))
        self.start.setText(_translate("MainWindow", "开始"))
        self.stop.setText(_translate("MainWindow", "暂停"))
        self.label_4.setText(_translate("MainWindow", "入"))
        self.label_5.setText(_translate("MainWindow", "出"))
        self.label_2.setText(_translate("MainWindow", "在线人数"))
        self.clearNum.setText(_translate("MainWindow", "清除数据"))
        self.close.setText(_translate("MainWindow", "关闭系统"))
        self.openWebcam_back.setText(_translate("MainWindow", "打开后门摄像头"))
        self.open2.setText(_translate("MainWindow", "打开文件"))
        self.start2.setText(_translate("MainWindow", "开始"))
        self.stop2.setText(_translate("MainWindow", "暂停"))
        self.action2.setText(_translate("MainWindow", "2"))
        self.action3.setText(_translate("MainWindow", "3"))
