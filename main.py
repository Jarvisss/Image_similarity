import sys
from PyQt5 import QtWidgets
from gui import MainWindow

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
    pass