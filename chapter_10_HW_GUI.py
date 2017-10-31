'''
Program for chapter 10 HW in Design of Experiments
Dr. Daily, University of Tulsa October 2017

Created by Daniel Moses
Calculates Anova table and residual plots for linear regression
'''
import sys, os
from PyQt5.QtWidgets import(QMainWindow,
                            QWidget,
                            QApplication,
                            QPushButton,
                            QTableWidget,
                            QTableWidgetItem,
                            QGroupBox,
                            QVBoxLayout,
                            QGridLayout,
                            QSizePolicy,
                            QLabel)
from PyQt5.QtCore import (QCoreApplication,QObject)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(hspace=0.5)
        self.axes = self.fig.add_subplot(211)
        self.axes_resid = self.fig.add_subplot(212)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        #self.plot_2_tail_t(1,1,1,1)

        #FigureCanvas.mpl_connect(self, 'button_press_event', self.export)

    '''def export(self, event):
        filename = "ExportedGraph.pdf"
        self.fig.savefig(filename)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Saved a copy of the graphics window to {}".format(filename))
        # msg.setInformativeText("This is additional information")
        msg.setWindowTitle("Saved PDF File")
        msg.setDetailedText("The full path of the file is \n{}".format(os.path.abspath(os.getcwd())))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowModality(Qt.ApplicationModal)
        msg.exec_()
        print("Exported PDF file")'''

class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself frequently with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.axes.set_xlabel("X Label")
        self.axes.set_ylabel("Y Label")
        self.axes.set_title("Title")

        self.axes_resid.set_ylabel("Y Label")
        self.draw()
    def plot_data(self,x_array,y_array,x_label,y_label,title,reg = False,max_val = 0, min_val = 0):
        self.axes.cla()
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.set_title(title)
        self.axes.plot(x_array,y_array,'b*')
        #if reg:
        #    self.axes.plot([min(x_array),max(x_array)],[min_val,max_val])
        self.draw()

    def plot_regression(self,x_limits,y_limits,beta):
        self.axes.plot(x_limits,y_limits)

        midpoint_x = (x_limits[0]+x_limits[1])/2
        midpoint_y = (y_limits[0]+y_limits[1])/2
        regression_eq = ('y = {0:0.2f} +'.format(beta[0])+'{0:0.2f}*x'.format(beta[1]))
        self.axes.annotate(regression_eq ,xy =(midpoint_x/1.5,midpoint_y))
        self.draw()
    def plot_residuals(self,x,y):
        self.axes_resid.cla()
        self.axes_resid.set_ylabel('Residuals')
        self.axes_resid.plot(x,y,'b*')
        self.axes_resid.plot([min(x),max(x)],[0,0])


class Generate_Anova (QMainWindow) :
    def __init__(self):
        super().__init__()

        self.init_ui()
    def init_ui(self):
        self.setGeometry(200,200,1000,700)
        main_widget = QWidget()
        self.add_tablebox()   #creates data table with load button
        self.add_anova_table()
        self.graph_canvas = MyDynamicMplCanvas(main_widget, width=15, height=8, dpi=100)

        main_grid_layout = QGridLayout()
        main_grid_layout.addWidget(self.tablebox,0,0,3,3)
        main_grid_layout.addWidget(self.anova_table,4,0,6,9)
        main_grid_layout.addWidget(self.graph_canvas,0,4,3,9)

        main_widget.setLayout(main_grid_layout)

        self.setCentralWidget(main_widget)
        self.setWindowTitle('Chapter 10 HW: Daniel Moses')
        self.show()


    def add_tablebox(self):
        self.data_table = QTableWidget()
        load_button = QPushButton('Load Data', self)
        load_button.clicked.connect(self.load_data)

        self.tablebox = QGroupBox("Data")
        tablebox_layout = QVBoxLayout()
        tablebox_layout.addWidget(load_button)
        tablebox_layout.addWidget(self.data_table)
        self.tablebox.setLayout(tablebox_layout)

    def add_anova_table(self):

        self.calculate_anova_btn = QPushButton('Calculate Anova Values')
        self.calculate_anova_btn.clicked.connect(self.calculate_anova)
        # Headers
        header_SOV = QLabel('Source of Variation')
        header_SOS = QLabel('Sum of Squares')
        header_DOF = QLabel('Degrees of Freedom')
        header_MS = QLabel('Mean Square')
        header_F0 = QLabel('F Statistic')
        header_Reg = QLabel('Regression')
        header_Res  = QLabel('Residual')
        header_Tot = QLabel('Total')

        self.ss_r_label = QLabel('SSr')
        self.ss_e_label = QLabel('SSe')
        self.ss_t_label = QLabel('SSt')
        self.df_r_label = QLabel('k')
        self.df_e_label = QLabel('n-k-1')
        self.df_t_label = QLabel('n-1')
        self.ms_r_label = QLabel('MSr')
        self.ms_e_label = QLabel('MSe')
        self.f0_label = QLabel('MSr/MSe')
        self.p_val_label = QLabel('P Value:')
        self.ci_B_label = QLabel("95% CI for B:")

        self.anova_table = QGroupBox('ANOVA Table')
        anova_table_layout = QGridLayout()
        anova_table_layout.addWidget(self.calculate_anova_btn, 3,4)
        anova_table_layout.addWidget(self.p_val_label,2,4)
        anova_table_layout.addWidget(self.ci_B_label,3,3)

        anova_table_layout.addWidget(header_SOV,0,0)
        anova_table_layout.addWidget(header_SOS,0,1)
        anova_table_layout.addWidget(header_DOF,0,2)
        anova_table_layout.addWidget(header_MS,0,3)
        anova_table_layout.addWidget(header_F0,0,4)
        anova_table_layout.addWidget(header_Reg,1,0)
        anova_table_layout.addWidget(header_Res,2,0)
        anova_table_layout.addWidget(header_Tot,3,0)

        anova_table_layout.addWidget(self.ss_r_label,1,1)
        anova_table_layout.addWidget(self.ss_e_label,2,1)
        anova_table_layout.addWidget(self.ss_t_label,3,1)
        anova_table_layout.addWidget(self.df_r_label,1,2)
        anova_table_layout.addWidget(self.df_e_label,2,2)
        anova_table_layout.addWidget(self.df_t_label,3,2)
        anova_table_layout.addWidget(self.ms_r_label,1,3)
        anova_table_layout.addWidget(self.ms_e_label,2,3)
        anova_table_layout.addWidget(self.f0_label,1,4)
        self.anova_table.setLayout(anova_table_layout)

    def calculate_anova(self):
        #Calculate Beta
        x_vals = self.data_1

        y_t = np.array(self.data_2)
        y = np.matrix.transpose(y_t)
        x_t = np.array([[1 for i in range(len(self.data_1))], x_vals])
        x = np.matrix.transpose(x_t)
        xtxi = np.linalg.inv(np.dot(x_t, x))
        beta = np.dot(xtxi, (np.dot(x_t, y)))
        beta_t = np.transpose(beta)

        y_hat = np.dot(x, beta)
        rs = y - y_hat

        n = len(x_vals)
        ss_e = np.dot(y_t, y) - np.dot(np.dot(beta_t, x_t), y)
        k = len(beta) - 1
        ss_r = np.dot(np.dot(beta_t, x_t), y) - np.sum(y) ** 2 / len(y)
        df_e = n - k - 1
        ss_t = ss_e + ss_r
        df_t = n - 1
        ms_r = ss_r / k
        ms_e = ss_e / df_e
        f0 = ms_r / ms_e
        ppf = stats.f.ppf(0.95, k, df_e)
        p_f = stats.f.cdf(f0, k, df_e)

        # Equations for B interval 10.38 Montgomery Text
        p = len(beta)
        sigma_hat_sqrd = ss_e / (n - p)
        # print('simga_hat =',sigma_hat)
        t_B = stats.t.ppf(0.975, n - p)
        cjj = xtxi[1][1]
        se_B = np.sqrt((sigma_hat_sqrd) * cjj)
        beta_upper = beta[1] + t_B * se_B
        beta_lower = beta[1] - t_B * se_B

        self.ss_r_label.setText('SSr = {0:.3f}'.format(ss_r))
        self.ss_e_label.setText('SSe = {0:.3f}'.format(ss_e))
        self.ss_t_label.setText('SSt = {0:.3f}'.format(ss_t))
        self.df_r_label.setText('k = {0:0.0f}'.format(k))
        self.df_e_label.setText('n-k-1 = {0:.0f}'.format(df_e))
        self.df_t_label.setText('n-1 = %g'%df_t)
        self.ms_r_label.setText('MSr = {0:0.3f}'.format(ms_r))
        self.ms_e_label.setText('MSe = {0:0.3f}'.format(ms_e))
        self.f0_label.setText('MSr/MSe = {0:0.3f}'.format(f0))
        self.p_val_label.setText('P Value: {0:0.4f}'.format(p_f))
        self.ci_B_label.setText('95% CI for B: {0:0.3f} < B < '.format(beta_lower)+'{0:0.3f}'.format(beta_upper))

        lin_x = [min(x_vals), max(x_vals)]
        lin_y = [y_hat[0], y_hat[-1]]
        self.graph_canvas.plot_regression(lin_x,lin_y,beta)
        print('chicken')
        self.graph_canvas.plot_residuals(x_vals,rs)
        print('mo-chickens')
        #plt.plot(x_vals, y, 'b*')
        #plt.plot(lin_x, lin_y)
        #plt.show()
        #plt.plot(x_vals,rs,'b*')
        #plt.show()

    def load_data(self):
        # for this example, we'll hard code the file name.
        data_file_name = "10_1HW.csv"
        # for the assignment, have a dialog box provide the filename

        # check to see if the file exists and then load it
        print(os.getcwd())
        print(os.path.exists(data_file_name))
        if os.path.exists(data_file_name):
            header_row = 1
            # load data file into memory as a list of lines
            with open(data_file_name, 'r') as data_file:
                self.data_lines = data_file.readlines()

            print("Opened {}".format(data_file_name))
            print(self.data_lines[1:10])

            # Set the headers
            # parse the lines by stripping the newline character off the end
            # and then splitting them on commas.
            self.data_table.clear()
            self.data_table.setRowCount(0)
            data_table_columns = self.data_lines[header_row].strip().split(',')

            self.data_table.setColumnCount(len(data_table_columns))
            self.data_table.setHorizontalHeaderLabels(data_table_columns)

            # fill the table starting with the row after the header
            self.data_1 = []
            self.data_2 = []
            for row in range(header_row + 1, len(self.data_lines)):
                # parse the data in memory into a list
                row_values = self.data_lines[row].strip().split(',')
                # insert a new row
                current_row = self.data_table.rowCount()
                self.data_table.insertRow(current_row)

                # Populate the row with data

                for col in range(len(data_table_columns)):
                    try:
                        if col == 0:
                            self.data_1.append(float(row_values[col]))
                        else:
                            self.data_2.append(float(row_values[col]))
                    except:
                        pass
                    entry = QTableWidgetItem("{}".format(row_values[col]))
                    self.data_table.setItem(current_row, col, entry)
            print("Filled {} rows.".format(row))

            self.graph_canvas.plot_data(self.data_1,self.data_2,data_table_columns[0],data_table_columns[1],'10.1 Data',reg = False)
        else:
            print("File not found.")


if __name__ == '__main__':
    # Start the program this way according to https://stackoverflow.com/questions/40094086/python-kernel-dies-for-second-run-of-pyqt5-gui
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    execute = Generate_Anova()
    sys.exit(app.exec_())