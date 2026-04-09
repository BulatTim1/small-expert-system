#!/usr/bin/env python3
"""Малая экспертная система с GUI на tkinter и Байесовским выводом."""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import copy
import math


# ─── Модель данных ───────────────────────────────────────────────────────────

class ExpertSystem:
    """Модель экспертной системы: загрузка/сохранение JSON, Байесовский вывод."""

    def __init__(self):
        self.name = "Новая система"
        self.description = ""
        self.questions = []
        self.outcomes = []

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.questions = data.get("questions", [])
        self.outcomes = data.get("outcomes", [])
        # Убедимся, что ключи коэффициентов — строки
        for o in self.outcomes:
            o["coefficients"] = {str(k): v for k, v in o.get("coefficients", {}).items()}

    def save(self, path):
        data = {
            "name": self.name,
            "description": self.description,
            "questions": self.questions,
            "outcomes": self.outcomes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def compute_posteriors(self, answers):
        """Вычислить апостериорные вероятности по теореме Байеса.

        answers — dict {question_id_str: value} где value от -5 до 5.
        Возвращает список (name, probability) отсортированный по убыванию.
        """
        n = len(self.outcomes)
        if n == 0:
            return []

        # Априорные вероятности
        priors = []
        for o in self.outcomes:
            priors.append(o.get("apriori", 1.0 / n))

        # Логарифмическое правдоподобие для числовой стабильности
        log_posteriors = [math.log(max(p, 1e-15)) for p in priors]

        for qid_str, answer in answers.items():
            p_yes = (answer + 5) / 10.0  # Нормализуем ответ в [0, 1]
            for i, o in enumerate(self.outcomes):
                c = o.get("coefficients", {}).get(qid_str, 0.5)
                # Правдоподобие: P(answer | H_i) = c * p_yes + (1 - c) * (1 - p_yes)
                likelihood = c * p_yes + (1.0 - c) * (1.0 - p_yes)
                likelihood = max(likelihood, 1e-15)  # Защита от log(0)
                log_posteriors[i] += math.log(likelihood)

        # Нормализация через log-sum-exp
        max_lp = max(log_posteriors)
        posteriors = [math.exp(lp - max_lp) for lp in log_posteriors]
        total = sum(posteriors)
        if total > 0:
            posteriors = [p / total for p in posteriors]

        result = [(self.outcomes[i]["name"], posteriors[i]) for i in range(n)]
        result.sort(key=lambda x: x[1], reverse=True)
        return result


# ─── Диалоги ─────────────────────────────────────────────────────────────────

class QuestionDialog(tk.Toplevel):
    """Диалог добавления/редактирования вопроса."""

    def __init__(self, parent, question=None):
        super().__init__(parent)
        self.title("Вопрос")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ttk.Label(self, text="ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.id_var = tk.StringVar(value=str(question["id"]) if question else "")
        ttk.Entry(self, textvariable=self.id_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self, text="Текст вопроса:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.text_var = tk.StringVar(value=question["text"] if question else "")
        ttk.Entry(self, textvariable=self.text_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.transient(parent)
        self.wait_window()

    def _ok(self):
        try:
            qid = int(self.id_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "ID должен быть целым числом.", parent=self)
            return
        text = self.text_var.get().strip()
        if not text:
            messagebox.showerror("Ошибка", "Текст вопроса не может быть пустым.", parent=self)
            return
        self.result = {"id": qid, "text": text}
        self.destroy()


class OutcomeDialog(tk.Toplevel):
    """Диалог добавления/редактирования исхода с коэффициентами."""

    def __init__(self, parent, questions, outcome=None):
        super().__init__(parent)
        self.title("Исход")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ttk.Label(self, text="Название:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.name_var = tk.StringVar(value=outcome["name"] if outcome else "")
        ttk.Entry(self, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self, text="Априорная вероятность:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.apriori_var = tk.StringVar(value=str(outcome.get("apriori", 0.5)) if outcome else "0.5")
        ttk.Entry(self, textvariable=self.apriori_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Коэффициенты
        coeff_frame = ttk.LabelFrame(self, text="Коэффициенты P(E=да | H) — от 0 до 1")
        coeff_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.coeff_vars = {}
        existing_coeffs = outcome.get("coefficients", {}) if outcome else {}
        for i, q in enumerate(questions):
            qid_str = str(q["id"])
            ttk.Label(coeff_frame, text=f"Q{q['id']}: {q['text'][:40]}").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            var = tk.StringVar(value=str(existing_coeffs.get(qid_str, "0.5")))
            ttk.Entry(coeff_frame, textvariable=var, width=8).grid(row=i, column=1, padx=5, pady=2)
            self.coeff_vars[qid_str] = var

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.transient(parent)
        self.wait_window()

    def _ok(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Название не может быть пустым.", parent=self)
            return
        try:
            apriori = float(self.apriori_var.get())
            if not (0 < apriori <= 1):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Априорная вероятность — число от 0 до 1.", parent=self)
            return

        coefficients = {}
        for qid_str, var in self.coeff_vars.items():
            try:
                val = float(var.get())
                if not (0 <= val <= 1):
                    raise ValueError
                coefficients[qid_str] = val
            except ValueError:
                messagebox.showerror("Ошибка", f"Коэффициент для Q{qid_str} должен быть числом от 0 до 1.", parent=self)
                return

        self.result = {"name": name, "apriori": apriori, "coefficients": coefficients}
        self.destroy()


# ─── Вкладка «Редактор» ─────────────────────────────────────────────────────

class EditorTab(ttk.Frame):
    """Вкладка для редактирования экспертной системы."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # Название и описание
        info_frame = ttk.LabelFrame(self, text="Информация о системе")
        info_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(info_frame, text="Название:").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.name_var, width=60).grid(row=0, column=1, padx=5, pady=3, sticky="w")

        ttk.Label(info_frame, text="Описание:").grid(row=1, column=0, padx=5, pady=3, sticky="ne")
        self.desc_text = tk.Text(info_frame, width=60, height=3)
        self.desc_text.grid(row=1, column=1, padx=5, pady=3, sticky="w")

        # Вопросы
        q_frame = ttk.LabelFrame(self, text="Вопросы")
        q_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.q_tree = ttk.Treeview(q_frame, columns=("id", "text"), show="headings", height=5)
        self.q_tree.heading("id", text="ID")
        self.q_tree.heading("text", text="Текст вопроса")
        self.q_tree.column("id", width=50, anchor="center")
        self.q_tree.column("text", width=500)
        self.q_tree.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

        q_scroll = ttk.Scrollbar(q_frame, orient="vertical", command=self.q_tree.yview)
        q_scroll.pack(side="left", fill="y")
        self.q_tree.configure(yscrollcommand=q_scroll.set)

        q_btn = ttk.Frame(q_frame)
        q_btn.pack(side="left", padx=5, pady=5)
        ttk.Button(q_btn, text="Добавить", command=self._add_question).pack(fill="x", pady=2)
        ttk.Button(q_btn, text="Редактировать", command=self._edit_question).pack(fill="x", pady=2)
        ttk.Button(q_btn, text="Удалить", command=self._del_question).pack(fill="x", pady=2)

        # Исходы
        o_frame = ttk.LabelFrame(self, text="Исходы (гипотезы)")
        o_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.o_tree = ttk.Treeview(o_frame, columns=("name", "apriori"), show="headings", height=5)
        self.o_tree.heading("name", text="Название")
        self.o_tree.heading("apriori", text="P(H)")
        self.o_tree.column("name", width=300)
        self.o_tree.column("apriori", width=80, anchor="center")
        self.o_tree.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

        o_scroll = ttk.Scrollbar(o_frame, orient="vertical", command=self.o_tree.yview)
        o_scroll.pack(side="left", fill="y")
        self.o_tree.configure(yscrollcommand=o_scroll.set)

        o_btn = ttk.Frame(o_frame)
        o_btn.pack(side="left", padx=5, pady=5)
        ttk.Button(o_btn, text="Добавить", command=self._add_outcome).pack(fill="x", pady=2)
        ttk.Button(o_btn, text="Редактировать", command=self._edit_outcome).pack(fill="x", pady=2)
        ttk.Button(o_btn, text="Удалить", command=self._del_outcome).pack(fill="x", pady=2)

    def load_from_model(self):
        es = self.app.expert_system
        self.name_var.set(es.name)
        self.desc_text.delete("1.0", "end")
        self.desc_text.insert("1.0", es.description)
        self._refresh_questions()
        self._refresh_outcomes()

    def save_to_model(self):
        es = self.app.expert_system
        es.name = self.name_var.get()
        es.description = self.desc_text.get("1.0", "end").strip()

    def _refresh_questions(self):
        self.q_tree.delete(*self.q_tree.get_children())
        for q in self.app.expert_system.questions:
            self.q_tree.insert("", "end", values=(q["id"], q["text"]))

    def _refresh_outcomes(self):
        self.o_tree.delete(*self.o_tree.get_children())
        for o in self.app.expert_system.outcomes:
            self.o_tree.insert("", "end", values=(o["name"], f"{o.get('apriori', 0.5):.2f}"))

    def _add_question(self):
        dlg = QuestionDialog(self)
        if dlg.result:
            self.app.expert_system.questions.append(dlg.result)
            self._refresh_questions()

    def _edit_question(self):
        sel = self.q_tree.selection()
        if not sel:
            return
        idx = self.q_tree.index(sel[0])
        q = self.app.expert_system.questions[idx]
        dlg = QuestionDialog(self, question=q)
        if dlg.result:
            self.app.expert_system.questions[idx] = dlg.result
            self._refresh_questions()

    def _del_question(self):
        sel = self.q_tree.selection()
        if not sel:
            return
        idx = self.q_tree.index(sel[0])
        del self.app.expert_system.questions[idx]
        self._refresh_questions()

    def _add_outcome(self):
        dlg = OutcomeDialog(self, self.app.expert_system.questions)
        if dlg.result:
            self.app.expert_system.outcomes.append(dlg.result)
            self._refresh_outcomes()

    def _edit_outcome(self):
        sel = self.o_tree.selection()
        if not sel:
            return
        idx = self.o_tree.index(sel[0])
        o = self.app.expert_system.outcomes[idx]
        dlg = OutcomeDialog(self, self.app.expert_system.questions, outcome=o)
        if dlg.result:
            self.app.expert_system.outcomes[idx] = dlg.result
            self._refresh_outcomes()

    def _del_outcome(self):
        sel = self.o_tree.selection()
        if not sel:
            return
        idx = self.o_tree.index(sel[0])
        del self.app.expert_system.outcomes[idx]
        self._refresh_outcomes()


# ─── Вкладка «Консультация» ─────────────────────────────────────────────────

class ConsultationTab(ttk.Frame):
    """Вкладка для проведения консультации с экспертной системой."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.current_q_idx = 0
        self.answers = {}
        self._build_ui()

    def _build_ui(self):
        # Левая панель — вопросы и слайдер
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.start_btn = ttk.Button(left, text="Начать консультацию", command=self._start)
        self.start_btn.pack(pady=10)

        self.question_label = ttk.Label(left, text="", font=("Arial", 14), wraplength=400, justify="center")
        self.question_label.pack(pady=20)

        self.slider_var = tk.IntVar(value=0)
        self.slider = tk.Scale(left, from_=-5, to=5, orient="horizontal", variable=self.slider_var,
                               length=300, tickinterval=1, command=self._on_slider_change)
        self.slider.pack(pady=10)
        self.slider.pack_forget()

        self.slider_label = ttk.Label(left, text="", font=("Arial", 11))
        self.slider_label.pack()

        self.answer_btn = ttk.Button(left, text="Ответить", command=self._answer)
        self.answer_btn.pack(pady=10)
        self.answer_btn.pack_forget()

        # Правая панель — вероятности
        right = ttk.LabelFrame(self, text="Вероятности исходов")
        right.pack(side="right", fill="both", padx=5, pady=5, ipadx=10)

        self.prob_tree = ttk.Treeview(right, columns=("name", "prob"), show="headings", height=10)
        self.prob_tree.heading("name", text="Исход")
        self.prob_tree.heading("prob", text="P(H|E)")
        self.prob_tree.column("name", width=180)
        self.prob_tree.column("prob", width=80, anchor="center")
        self.prob_tree.pack(fill="both", expand=True, padx=5, pady=5)

    def _start(self):
        es = self.app.expert_system
        if not es.questions or not es.outcomes:
            messagebox.showwarning("Внимание", "Система должна содержать вопросы и исходы.")
            return
        self.current_q_idx = 0
        self.answers = {}
        self.slider.pack(pady=10)
        self.answer_btn.pack(pady=10)
        self.start_btn.config(state="disabled")
        self._show_question()
        self._update_probabilities()

    def _show_question(self):
        es = self.app.expert_system
        q = es.questions[self.current_q_idx]
        self.question_label.config(text=f"Вопрос {self.current_q_idx + 1}/{len(es.questions)}:\n\n{q['text']}")
        self.slider_var.set(0)
        self.slider_label.config(text="Текущий ответ: 0")

    def _on_slider_change(self, val):
        self.slider_label.config(text=f"Текущий ответ: {int(float(val))}")
        # Превью вероятностей с текущим ответом
        self._update_probabilities(preview_answer=int(float(val)))

    def _update_probabilities(self, preview_answer=None):
        es = self.app.expert_system
        answers = dict(self.answers)
        if preview_answer is not None and self.current_q_idx < len(es.questions):
            qid = str(es.questions[self.current_q_idx]["id"])
            answers[qid] = preview_answer

        posteriors = es.compute_posteriors(answers)
        self.prob_tree.delete(*self.prob_tree.get_children())
        for name, prob in posteriors:
            self.prob_tree.insert("", "end", values=(name, f"{prob * 100:.1f}%"))

    def _answer(self):
        es = self.app.expert_system
        q = es.questions[self.current_q_idx]
        self.answers[str(q["id"])] = self.slider_var.get()
        self.current_q_idx += 1

        if self.current_q_idx < len(es.questions):
            self._show_question()
            self._update_probabilities()
        else:
            self._finish()

    def _finish(self):
        es = self.app.expert_system
        posteriors = es.compute_posteriors(self.answers)
        self._update_probabilities()

        self.slider.pack_forget()
        self.answer_btn.pack_forget()
        self.slider_label.config(text="")
        self.start_btn.config(state="normal")

        if posteriors:
            best_name, best_prob = posteriors[0]
            result_text = f"Наиболее вероятный исход:\n\n{best_name}\n(вероятность: {best_prob * 100:.1f}%)"
            self.question_label.config(text=result_text)
            messagebox.showinfo("Результат консультации",
                                f"Наиболее вероятный исход: {best_name}\n"
                                f"Вероятность: {best_prob * 100:.1f}%")
        else:
            self.question_label.config(text="Нет данных для вывода.")


# ─── Вкладка «Дерево решений» ───────────────────────────────────────────────

class TreeTab(ttk.Frame):
    """Вкладка для визуализации дерева решений."""

    NODE_W = 140
    NODE_H = 40
    H_GAP = 20
    V_GAP = 60

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=5, pady=5)
        ttk.Button(toolbar, text="Построить дерево", command=self._draw_tree).pack(side="left")

        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    def _draw_tree(self):
        es = self.app.expert_system
        if not es.questions or not es.outcomes:
            messagebox.showwarning("Внимание", "Система должна содержать вопросы и исходы.")
            return

        n_q = len(es.questions)
        if n_q > 6:
            if not messagebox.askyesno("Предупреждение",
                                       f"Вопросов: {n_q}. Дерево будет содержать {2 ** n_q} листьев.\n"
                                       "Построение может занять время. Продолжить?"):
                return

        self.canvas.delete("all")

        # Вычисляем размеры дерева
        n_leaves = 2 ** n_q
        tree_w = n_leaves * (self.NODE_W + self.H_GAP)
        tree_h = (n_q + 1) * (self.NODE_H + self.V_GAP) + 40

        root_x = tree_w / 2
        root_y = 30

        self._draw_node(es, root_x, root_y, 0, {}, tree_w / 2)

        self.canvas.configure(scrollregion=(0, 0, tree_w, tree_h))

    def _draw_node(self, es, x, y, q_idx, answers, half_width):
        """Рекурсивно рисует узлы дерева."""
        if q_idx >= len(es.questions):
            # Лист — показать лучший исход
            posteriors = es.compute_posteriors(answers)
            if posteriors:
                best_name, best_prob = posteriors[0]
                text = f"{best_name}\n{best_prob * 100:.1f}%"
            else:
                text = "?"
            self.canvas.create_rectangle(x - self.NODE_W // 2, y,
                                         x + self.NODE_W // 2, y + self.NODE_H,
                                         fill="#90EE90", outline="#2E8B57", width=2)
            self.canvas.create_text(x, y + self.NODE_H // 2, text=text, font=("Arial", 8),
                                    width=self.NODE_W - 10, justify="center")
            return

        q = es.questions[q_idx]
        # Рисуем узел вопроса
        self.canvas.create_rectangle(x - self.NODE_W // 2, y,
                                     x + self.NODE_W // 2, y + self.NODE_H,
                                     fill="#87CEEB", outline="#4682B4", width=2)
        self.canvas.create_text(x, y + self.NODE_H // 2,
                                text=f"Q{q['id']}: {q['text'][:18]}",
                                font=("Arial", 8), width=self.NODE_W - 10, justify="center")

        child_y = y + self.NODE_H + self.V_GAP
        child_half = half_width / 2

        # Левая ветка: ответ = -5
        left_x = x - child_half
        left_answers = dict(answers)
        left_answers[str(q["id"])] = -5
        self.canvas.create_line(x, y + self.NODE_H, left_x, child_y, fill="#555")
        self.canvas.create_text((x + left_x) / 2 - 10, (y + self.NODE_H + child_y) / 2,
                                text="-5", font=("Arial", 8, "bold"), fill="#CC0000")
        self._draw_node(es, left_x, child_y, q_idx + 1, left_answers, child_half)

        # Правая ветка: ответ = +5
        right_x = x + child_half
        right_answers = dict(answers)
        right_answers[str(q["id"])] = 5
        self.canvas.create_line(x, y + self.NODE_H, right_x, child_y, fill="#555")
        self.canvas.create_text((x + right_x) / 2 + 10, (y + self.NODE_H + child_y) / 2,
                                text="+5", font=("Arial", 8, "bold"), fill="#006600")
        self._draw_node(es, right_x, child_y, q_idx + 1, right_answers, child_half)


# ─── Главное приложение ──────────────────────────────────────────────────────

class ExpertSystemApp(tk.Tk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.title("Малая экспертная система")
        self.geometry("900x650")

        self.expert_system = ExpertSystem()
        self.current_file = None

        self._build_menu()
        self._build_tabs()

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Новая система", command=self._new_system)
        file_menu.add_command(label="Открыть...", command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить", command=self._save_file)
        file_menu.add_command(label="Сохранить как...", command=self._save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)
        self.config(menu=menubar)

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.editor_tab = EditorTab(self.notebook, self)
        self.consult_tab = ConsultationTab(self.notebook, self)
        self.tree_tab = TreeTab(self.notebook, self)

        self.notebook.add(self.editor_tab, text="Редактор")
        self.notebook.add(self.consult_tab, text="Консультация")
        self.notebook.add(self.tree_tab, text="Дерево решений")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        # При переключении на редактор — сохраняем модель, потом обновляем UI
        current = self.notebook.index(self.notebook.select())
        if current == 0:
            self.editor_tab.load_from_model()

    def _new_system(self):
        self.expert_system = ExpertSystem()
        self.current_file = None
        self.title("Малая экспертная система")
        self.editor_tab.load_from_model()

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        if path:
            try:
                self.expert_system.load(path)
                self.current_file = path
                self.title(f"Малая экспертная система — {path}")
                self.editor_tab.load_from_model()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")

    def _save_file(self):
        self.editor_tab.save_to_model()
        if self.current_file:
            try:
                self.expert_system.save(self.current_file)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
        else:
            self._save_file_as()

    def _save_file_as(self):
        self.editor_tab.save_to_model()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        if path:
            try:
                self.expert_system.save(path)
                self.current_file = path
                self.title(f"Малая экспертная система — {path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")


def main():
    app = ExpertSystemApp()
    app.mainloop()


if __name__ == "__main__":
    main()
