import sys
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QTableWidget,
    QComboBox, QDoubleSpinBox, QFileDialog
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class FuzzyController2Input:
    def __init__(self):
        self.mf_params = {
            "angle": {
                "neg": [-30, -30, 0],
                "zero": [-10, 0, 10],
                "pos": [0, 30, 30],
            },
            "acc": {
                "neg": [-100, -100, 0],
                "zero": [-20, 0, 20],
                "pos": [0, 100, 100],
            },
            "pwm": {
                "low": [0, 0, 128],
                "mid": [64, 128, 192],
                "high": [128, 255, 255],
            },
        }

        self.rule_data = [
            ("zero", "zero", "mid"),
            ("pos", "pos", "low"),
            ("neg", "neg", "low"),
            ("pos", "zero", "low"),
            ("neg", "zero", "low"),
            ("zero", "pos", "high"),
            ("zero", "neg", "high"),
        ]

        self.build_system()

    def build_system(self):
        self.angle = ctrl.Antecedent(np.arange(-30, 31, 1), "angle")
        self.acc = ctrl.Antecedent(np.arange(-100, 101, 1), "acc")
        self.pwm = ctrl.Consequent(np.arange(0, 256, 1), "pwm")

        for label, params in self.mf_params["angle"].items():
            self.angle[label] = fuzz.trimf(self.angle.universe, params)

        for label, params in self.mf_params["acc"].items():
            self.acc[label] = fuzz.trimf(self.acc.universe, params)

        for label, params in self.mf_params["pwm"].items():
            self.pwm[label] = fuzz.trimf(self.pwm.universe, params)

        rules = []
        for angle_label, acc_label, pwm_label in self.rule_data:
            rules.append(
                ctrl.Rule(
                    self.angle[angle_label] & self.acc[acc_label],
                    self.pwm[pwm_label],
                )
            )

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def update_rules(self, rule_data):
        self.rule_data = rule_data
        self.build_system()

    def update_mf(self, var_name, mf_name, params):
        params = sorted(params)
        self.mf_params[var_name][mf_name] = params
        self.build_system()

    def compute(self, angle_val, acc_val):
        self.sim = ctrl.ControlSystemSimulation(self.system)
        self.sim.input["angle"] = angle_val
        self.sim.input["acc"] = acc_val
        self.sim.compute()
        return self.sim.output["pwm"]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class RuleEditor(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Angle", "Accelerate", "PWM"])

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Rule")
        self.delete_btn = QPushButton("Delete Selected Rule")
        self.apply_btn = QPushButton("Apply Rules")

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.apply_btn)

        layout.addWidget(QLabel("Rule Editor"))
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        self.add_btn.clicked.connect(lambda: self.add_rule())
        self.delete_btn.clicked.connect(self.delete_rule)
        self.apply_btn.clicked.connect(self.apply_rules)

        self.load_rules()

    def create_combo(self, items, current):
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(current)
        return combo

    def load_rules(self):
        self.table.setRowCount(0)
        for angle, acc, pwm in self.controller.rule_data:
            self.add_rule(angle, acc, pwm)

    def add_rule(self, angle="zero", acc="zero", pwm="mid"):
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setCellWidget(row, 0, self.create_combo(["neg", "zero", "pos"], angle))
        self.table.setCellWidget(row, 1, self.create_combo(["neg", "zero", "pos"], acc))
        self.table.setCellWidget(row, 2, self.create_combo(["low", "mid", "high"], pwm))

    def delete_rule(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def apply_rules(self):
        rule_data = []

        for row in range(self.table.rowCount()):
            angle = self.table.cellWidget(row, 0).currentText()
            acc = self.table.cellWidget(row, 1).currentText()
            pwm = self.table.cellWidget(row, 2).currentText()
            rule_data.append((angle, acc, pwm))

        self.controller.update_rules(rule_data)


class MembershipEditor(QWidget):
    def __init__(self, controller, redraw_callback):
        super().__init__()
        self.controller = controller
        self.redraw_callback = redraw_callback

        layout = QVBoxLayout(self)

        self.var_combo = QComboBox()
        self.var_combo.addItems(["angle", "acc", "pwm"])

        self.mf_combo = QComboBox()

        self.a_spin = QDoubleSpinBox()
        self.b_spin = QDoubleSpinBox()
        self.c_spin = QDoubleSpinBox()

        for spin in [self.a_spin, self.b_spin, self.c_spin]:
            spin.setRange(-1000, 1000)
            spin.setDecimals(2)

        self.apply_btn = QPushButton("Apply MF")

        layout.addWidget(QLabel("Membership Function Editor"))
        layout.addWidget(QLabel("Variable"))
        layout.addWidget(self.var_combo)
        layout.addWidget(QLabel("Membership Function"))
        layout.addWidget(self.mf_combo)

        layout.addWidget(QLabel("a"))
        layout.addWidget(self.a_spin)
        layout.addWidget(QLabel("b"))
        layout.addWidget(self.b_spin)
        layout.addWidget(QLabel("c"))
        layout.addWidget(self.c_spin)

        layout.addWidget(self.apply_btn)

        self.var_combo.currentTextChanged.connect(self.update_mf_combo)
        self.mf_combo.currentTextChanged.connect(self.load_params)
        self.apply_btn.clicked.connect(self.apply_mf)

        self.update_mf_combo()

    def update_mf_combo(self):
        var_name = self.var_combo.currentText()
        self.mf_combo.clear()
        self.mf_combo.addItems(list(self.controller.mf_params[var_name].keys()))
        self.load_params()

    def load_params(self):
        var_name = self.var_combo.currentText()
        mf_name = self.mf_combo.currentText()

        if not mf_name:
            return

        a, b, c = self.controller.mf_params[var_name][mf_name]

        self.a_spin.setValue(a)
        self.b_spin.setValue(b)
        self.c_spin.setValue(c)

    def apply_mf(self):
        var_name = self.var_combo.currentText()
        mf_name = self.mf_combo.currentText()

        params = [
            self.a_spin.value(),
            self.b_spin.value(),
            self.c_spin.value(),
        ]

        self.controller.update_mf(var_name, mf_name, params)
        self.redraw_callback()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fuzzy PWM Controller with Rule and MF Editor")
        self.resize(1200, 700)

        self.controller = FuzzyController2Input()

        self.dragging_point = None

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()

        self.angle_label = QLabel("Angle: 0 deg")
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-30, 30)
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self.update_labels)

        self.acc_label = QLabel("Accelerate: 0 m/s^2")
        self.acc_slider = QSlider(Qt.Horizontal)
        self.acc_slider.setRange(-100, 100)
        self.acc_slider.setValue(0)
        self.acc_slider.valueChanged.connect(self.update_labels)

        self.run_btn = QPushButton("Compute PWM")
        self.run_btn.clicked.connect(self.run_fuzzy)

        self.output_label = QLabel("PWM Output: -")

        self.export_c_btn = QPushButton("Export C")
        self.export_c_btn.clicked.connect(self.export_c_code)

        left_layout.addWidget(self.angle_label)
        left_layout.addWidget(self.angle_slider)
        left_layout.addWidget(self.acc_label)
        left_layout.addWidget(self.acc_slider)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.output_label)
        left_layout.addWidget(self.export_c_btn)

        self.rule_editor = RuleEditor(self.controller)
        self.mf_editor = MembershipEditor(self.controller, self.update_plot)

        left_layout.addWidget(self.rule_editor)
        left_layout.addWidget(self.mf_editor)

        self.canvas = MplCanvas()

        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(self.canvas, 3)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.update_plot()

    def update_labels(self):
        self.angle_label.setText(f"Angle: {self.angle_slider.value()} deg")
        self.acc_label.setText(f"Accelerate: {self.acc_slider.value()} m/s^2")

    def run_fuzzy(self):
        angle = self.angle_slider.value()
        acc = self.acc_slider.value()

        pwm = self.controller.compute(angle, acc)

        self.output_label.setText(f"PWM Output: {pwm:.1f}")
        self.update_plot()

    def get_selected_mf(self):
        var_name = self.mf_editor.var_combo.currentText()
        mf_name = self.mf_editor.mf_combo.currentText()
        return var_name, mf_name

    def get_universe(self, var_name):
        if var_name == "angle":
            return self.controller.angle.universe
        elif var_name == "acc":
            return self.controller.acc.universe
        else:
            return self.controller.pwm.universe

    def update_plot(self):
        ax = self.canvas.ax
        ax.clear()

        var_name, mf_name = self.get_selected_mf()
        universe = self.get_universe(var_name)

        for label, params in self.controller.mf_params[var_name].items():
            y = fuzz.trimf(universe, params)
            ax.plot(universe, y, label=label)

            if label == mf_name:
                points_x = params
                points_y = [0, 1, 0]
                ax.scatter(points_x, points_y, s=80)

        ax.set_title(f"{var_name} membership functions")
        ax.set_xlabel(var_name)
        ax.set_ylabel("Membership")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(True)

        self.canvas.draw()

    def on_press(self, event):
        if event.xdata is None or event.ydata is None:
            return

        var_name, mf_name = self.get_selected_mf()
        params = self.controller.mf_params[var_name][mf_name]

        points = [(params[0], 0), (params[1], 1), (params[2], 0)]

        min_dist = 999999
        nearest_index = None

        for i, (px, py) in enumerate(points):
            dist = abs(event.xdata - px) + abs(event.ydata - py)
            if dist < min_dist:
                min_dist = dist
                nearest_index = i

        if min_dist < 5:
            self.dragging_point = nearest_index

    def on_motion(self, event):
        if self.dragging_point is None:
            return
        if event.xdata is None:
            return

        var_name, mf_name = self.get_selected_mf()
        params = self.controller.mf_params[var_name][mf_name]

        new_params = params[:]
        new_params[self.dragging_point] = event.xdata
        new_params = sorted(new_params)

        self.controller.mf_params[var_name][mf_name] = new_params

        self.mf_editor.a_spin.setValue(new_params[0])
        self.mf_editor.b_spin.setValue(new_params[1])
        self.mf_editor.c_spin.setValue(new_params[2])

        self.controller.build_system()
        self.update_plot()

    def on_release(self, event):
        self.dragging_point = None

    def export_c_code(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")

        if not folder:
            return

        c_code, h_code = self.generate_c_code()

        with open(f"{folder}/fuzzy_export.c", "w", encoding="utf-8") as f:
            f.write(c_code)

        with open(f"{folder}/fuzzy_export.h", "w", encoding="utf-8") as f:
            f.write(h_code)

        self.output_label.setText("Exported: fuzzy_export.c / fuzzy_export.h")

    def generate_c_code(self):
        mf = self.controller.mf_params
        rules = self.controller.rule_data

        h_code = """#ifndef FUZZY_EXPORT_H
    #define FUZZY_EXPORT_H

    float fuzzy_compute(float angle, float acc);

    #endif
    """

        c_code = """#include "fuzzy_export.h"

    float trimf(float x, float a, float b, float c)
    {
        if (x <= a) return 0.0f;
        if (x >= c) return 0.0f;
        if (x == b) return 1.0f;

        if (x < b)
        {
            if (b == a) return 1.0f;
            return (x - a) / (b - a);
        }
        else
        {
            if (c == b) return 1.0f;
            return (c - x) / (c - b);
        }
    }

    float clamp(float x, float min, float max)
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    """

        # メンバーシップ関数をC関数として出力
        for var_name, funcs in mf.items():
            for mf_name, params in funcs.items():
                a, b, c = params
                c_code += f"""float {var_name}_{mf_name}(float x)
    {{
        return trimf(x, {a:.6f}f, {b:.6f}f, {c:.6f}f);
    }}

    """

        # PWM代表値
        pwm_centers = {}
        for name, params in mf["pwm"].items():
            pwm_centers[name] = params[1]

        c_code += """float fuzzy_compute(float angle, float acc)
    {
        float numerator = 0.0f;
        float denominator = 0.0f;
        float w = 0.0f;

    """

        # ルール出力
        for angle_label, acc_label, pwm_label in rules:
            center = pwm_centers[pwm_label]

            c_code += f"""    // IF angle is {angle_label} AND acc is {acc_label} THEN pwm is {pwm_label}
        w = angle_{angle_label}(angle);
        if (acc_{acc_label}(acc) < w) w = acc_{acc_label}(acc);

        numerator += w * {center:.6f}f;
        denominator += w;

    """

        c_code += """    if (denominator == 0.0f)
        {
            return 0.0f;
        }

        return clamp(numerator / denominator, 0.0f, 255.0f);
    }
    """

        return c_code, h_code


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())