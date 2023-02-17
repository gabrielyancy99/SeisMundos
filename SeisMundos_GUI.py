from PyQt6.QtCore import QSize, Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QPixmap, QDesktopServices, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QWidget, QToolTip, QSlider, QComboBox
import sys
import global_word_cluster as gwc
import os

class SaveButton(QPushButton):
    def __init__(self, graphname):
        super().__init__("Save Graph", graphname)
        self.graphname = None
        self.clicked.connect(self.save_file)
        self.message = 'Saves a copy to local directory with timestamp.'

    def save_file(self, event):
        os.system('cp '+self.graphname+'.png '+self.graphname+str(os.path.getmtime(self.graphname+'.png'))+'.png')

    def enterEvent(self, event):
        QToolTip.showText(QCursor.pos(), self.message)

    def leaveEvent(self, event):
        QToolTip.hideText()


class FileLabel(QLabel):
    def __init__(self, filename):
        super().__init__()
        self.filename = None
        self.message = 'Double click to reopen graph in seperate window.'
        self.graph = None

    def enterEvent(self, event):
        QToolTip.showText(QCursor.pos(), self.message)

    def leaveEvent(self, event):
        QToolTip.hideText()

    def mouseDoubleClickEvent(self, event):
        # QDesktopServices.openUrl(QUrl.fromLocalFile(self.filename))
        self.graph.show()

class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.aspect_ratio = 12 / 20
        self.setFixedSize(QSize(1000,1000*self.aspect_ratio))
        self.resize(1000,1000*self.aspect_ratio)  # set initial size
        self.setMinimumSize(350,350*self.aspect_ratio)  # set minimum size
        self.setMaximumSize(1500,1500*self.aspect_ratio)  # set maximum size
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen_geometry.topLeft())

    def resizeEvent(self, event):
        # Calculate the new height based on the aspect ratio and the new width
        new_width = event.size().width()
        new_height = int(new_width * self.aspect_ratio)

        # Enforce the aspect ratio by setting the new height to maintain it
        if new_height != event.size().height():
            new_size = event.size()
            new_size.setHeight(new_height)
            self.resize(new_size)

class MainWindow(QMainWindow):
    graph_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.search_term = None
        self.method_name = 'distance'
        self.cluster_number = 6
        self.w = None
        self.v = None
        self.u = None
        self.t = None

        ##### Set Up Layouts
        # self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("SeisMundos - Etymology Approximation with Levenshtein Distance")
        self.setFixedSize(QSize(1000,550))
        self.resize(1000, 550)  # set initial size
        self.setMinimumSize(750, 500)  # set minimum size
        self.setMaximumSize(1500, 700)  # set maximum size
        self.layout=QVBoxLayout()
        self.graphlayout=QHBoxLayout()
        self.graphlayout2=QHBoxLayout()
        self.graphsall=QVBoxLayout()
        self.optionslayout=QHBoxLayout()
        self.clusterlayout=QVBoxLayout()
        self.methodlayout=QVBoxLayout()

        ##### Set Up Main Header
        description = QLabel("See the Relationship Between Languages Across the World!")
        text_font = description.font()
        text_font.setPointSize(20)
        description.setFont(text_font)
        description.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        ##### Set Up Entry Field For Term
        term_entry_description = QLabel("What is the word you would like to analyze?")
        alt_text_font = term_entry_description.font()
        alt_text_font.setPointSize(14)
        term_entry_description.setFont(text_font)
        term_entry_description.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        term_entry = QLineEdit()
        term_entry.setPlaceholderText("Enter the word in English here...")
        term_entry.textChanged.connect(self.new_search_term)
        term_entry.returnPressed.connect(self.the_button_was_pressed)
        
        ##### Set Up Entry Field For Cluster Number
        cluster_entry_description = QLabel("How many clusters do you wnat to make overall?")
        cluster_entry_description.setFont(alt_text_font)
        cluster_entry_description.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # cluster_entry = QLineEdit()
        # cluster_entry.setPlaceholderText("Enter the number of clusters here (2-10)...")
        # cluster_entry.textChanged.connect(self.new_cluster_number)
        # cluster_entry.returnPressed.connect(self.the_button_was_pressed)

        cluster_entry = QSlider(Qt.Orientation.Horizontal, self)
        cluster_entry.setMinimum(2)
        cluster_entry.setMaximum(10)
        cluster_entry.setTickPosition(QSlider.TickPosition.TicksBelow)
        cluster_entry.setTickInterval(1)
        cluster_entry.setSingleStep(1)
        cluster_entry.setValue(6)
        cluster_entry.valueChanged.connect(self.new_cluster_number)
        # cluster_entry.returnPressed.connect(self.the_button_was_pressed)

        self.cluster_number_label = QLabel()
        self.cluster_number_label.setText("Clusters: "+str(self.cluster_number))
        self.cluster_number_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        ##### Set Up Entry Field For Method Name
        method_entry_description = QLabel("How do you want to measure similarity?")
        method_entry_description.setFont(alt_text_font)
        method_entry_description.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # method_entry = QLineEdit()
        # method_entry.setPlaceholderText("Enter either 'distance' or 'ratio'...")
        # method_entry.textChanged.connect(self.new_method_entry)
        # method_entry.returnPressed.connect(self.the_button_was_pressed)

        method_entry = QComboBox()
        method_entry.addItem('distance')
        method_entry.addItem('ratio')
        method_entry.textActivated[str].connect(self.new_method_entry)

        ##### Set Up Button
        self.button = QPushButton("Mundo Your Word")
        self.button.setMinimumSize(QSize(650,60))
        self.button.setStyleSheet('background-color: rgb(145, 41, 41); width: 90vw; height: 50vw;')
        self.button.clicked.connect(self.the_button_was_pressed)
        self.graph_signal.connect(self.the_graph_appeared)
        
        ##### Set Up Graphs
        self.graph1 = FileLabel(self)
        self.graph1.filename = 'world_ex.png'
        self.graph1.setScaledContents(True)
        self.graph1.setStyleSheet('width: 45vw; height: 40vw;')

        self.graph2 = FileLabel(self)
        self.graph2.filename = 'dendrogram_ex.png'
        self.graph2.setStyleSheet('width: 45vw; height: 40vw;')
        self.graph2.setScaledContents(True)

        self.graph3 = FileLabel(self)
        self.graph3.filename = 'europe_ex.png'
        self.graph3.setStyleSheet('width: 45vw; height: 40vw;')
        self.graph3.setScaledContents(True)

        self.graph4 = FileLabel(self)
        self.graph4.filename = 'asia_ex.png'
        self.graph4.setStyleSheet('width: 45vw; height: 40vw;')
        self.graph4.setScaledContents(True)

        ##### Build Up the Layouts
        self.graphlayout.addWidget(self.graph1)
        self.graphlayout.addWidget(self.graph2)

        graph_row1 = QWidget()
        graph_row1.setLayout(self.graphlayout)

        self.graphlayout2.addWidget(self.graph3)
        self.graphlayout2.addWidget(self.graph4)
        
        graph_row2 = QWidget()
        graph_row2.setLayout(self.graphlayout2)

        self.graphsall.addWidget(graph_row1)
        self.graphsall.addWidget(graph_row2)

        graphs = QWidget()
        graphs.setLayout(self.graphsall)

        self.clusterlayout.addWidget(cluster_entry_description)
        self.clusterlayout.addWidget(cluster_entry)
        self.clusterlayout.addWidget(self.cluster_number_label)

        cluster = QWidget()
        cluster.setLayout(self.clusterlayout)

        self.methodlayout.addWidget(method_entry_description)
        self.methodlayout.addWidget(method_entry)
        
        method = QWidget()
        method.setLayout(self.methodlayout)

        self.optionslayout.addWidget(cluster)
        self.optionslayout.addWidget(method)

        options = QWidget()
        options.setLayout(self.optionslayout)

        self.layout.addWidget(description)
        self.layout.addWidget(term_entry_description)
        self.layout.addWidget(term_entry)
        self.layout.addWidget(options)
        self.layout.addWidget(graphs)
        self.layout.addWidget(self.button)

        widget = QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.setStyleSheet('color: white; background-color: #222222;')

    def the_button_was_pressed(self):
        print("Pressed!")
        if self.search_term == None:
            print("Enter a value for the search term!")
        elif self.cluster_number == None:
            print("Enter a value for the cluster number!")
        elif self.method_name == None:
            print("Enter a value for the method name!")
        else:
            self.button.setText("Running..")
            self.button.setEnabled(False)
            print(self.search_term)
            # breakpoint()
            # print(self.sortedcounts)
            gwc.run_main(term=self.search_term, num_clust=int(self.cluster_number), method=self.method_name)
            self.graph_signal.emit()

    def new_search_term(self, s):
        self.search_term = s

    def new_cluster_number(self, s):
        self.cluster_number = s
        self.cluster_number_label.setText("Clusters: "+str(self.cluster_number))
        self.cluster_number_label.adjustSize()

    def new_method_entry(self, s):
        self.method_name = s

    def the_graph_appeared(self):
        pixmap = QPixmap('world_ex.png')
        self.graph1.setPixmap(pixmap)
        self.v = AnotherWindow()
        self.v.graph = QLabel()
        self.v.graph.setPixmap(pixmap)
        self.v.graph.setScaledContents(True)
        self.v.save = SaveButton(self)
        self.v.save.graphname = 'world_ex'
        self.v.layout.addWidget(self.v.save)
        self.v.layout.addWidget(self.v.graph)        
        self.v.setWindowTitle("World Map!")
        self.graph1.graph = self.v

        pixmap = QPixmap('dendrogram_ex.png')
        self.graph2.setPixmap(pixmap)
        self.button.setEnabled(True)
        self.button.setText("Search Another Term")
        self.w = AnotherWindow()
        self.w.graph = QLabel()
        self.w.graph.setPixmap(pixmap)
        self.w.graph.setScaledContents(True)
        self.w.save = SaveButton(self)
        self.w.save.graphname = 'dendrogram_ex'
        self.w.layout.addWidget(self.w.save)
        self.w.layout.addWidget(self.w.graph)
        self.w.setWindowTitle("Dendrogram!")
        self.graph2.graph = self.w

        pixmap = QPixmap('europe_ex.png')
        self.graph3.setPixmap(pixmap)
        self.u = AnotherWindow()
        self.u.graph = QLabel()
        self.u.graph.setPixmap(pixmap)
        self.u.graph.setScaledContents(True)
        self.u.save = SaveButton(self)
        self.u.save.graphname = 'europe_ex'
        self.u.layout.addWidget(self.u.save)
        self.u.layout.addWidget(self.u.graph)
        self.u.setWindowTitle("Map of Europe!")
        self.graph3.graph = self.u


        pixmap = QPixmap('asia_ex.png')
        self.graph4.setPixmap(pixmap)
        self.t = AnotherWindow()
        self.t.graph = QLabel()
        self.t.graph.setPixmap(pixmap)
        self.t.graph.setScaledContents(True)
        self.t.save = SaveButton(self)
        self.t.save.graphname = 'asia_ex'
        self.t.layout.addWidget(self.t.save)
        self.t.layout.addWidget(self.t.graph)
        self.t.setWindowTitle("Map of Asia!")
        self.graph4.graph = self.t

        self.t.show()
        self.u.show()
        self.v.show()
        self.w.show()
        self.resize(800, 1200)  # set initial size
        self.setMinimumSize(675, 750)  # set minimum size
        self.setMaximumSize(1000, 1500)  # set maximum size



app = QApplication(sys.argv)


window = MainWindow()
window.show()

sys.exit(app.exec())