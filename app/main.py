import sys

from app.bootstrap import bootstrap_project_paths
bootstrap_project_paths()

from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()