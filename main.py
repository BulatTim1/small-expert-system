#!/usr/bin/env python3
"""Малая экспертная система с GUI на tkinter и Байесовским выводом."""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# import copy
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

        # Априорные вероятности — равномерные (1/n)
        priors = [1.0 / n for _ in self.outcomes]

        # Логарифмическое правдоподобие для числовой стабильности
        log_posteriors = [math.log(max(p, 1e-15)) for p in priors]

        for qid_str, answer in answers.items():
            p_yes = float(answer)  # Значение ответа уже в диапазоне [0, 1]
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

_DEFAULT_ANSWERS = [{"text": "Да", "value": 1.0}, {"text": "Нет", "value": 0.0}]


class AnswerChoiceDialog(tk.Toplevel):
    """Диалог добавления варианта ответа для вопроса."""

    def __init__(self, parent, choice=None):
        super().__init__(parent)
        self.title("Вариант ответа")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ttk.Label(self, text="Текст ответа:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.text_var = tk.StringVar(value=choice["text"] if choice else "")
        ttk.Entry(self, textvariable=self.text_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self, text="Значение (0..1):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.value_var = tk.StringVar(value=str(choice["value"]) if choice else "0.5")
        ttk.Entry(self, textvariable=self.value_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.transient(parent)
        self.wait_window()

    def _ok(self):
        text = self.text_var.get().strip()
        if not text:
            messagebox.showerror("Ошибка", "Текст ответа не может быть пустым.", parent=self)
            return
        try:
            value = float(self.value_var.get())
            if not (0.0 <= value <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Значение должно быть числом от 0 до 1.", parent=self)
            return
        self.result = {"text": text, "value": value}
        self.destroy()


class QuestionDialog(tk.Toplevel):
    """Диалог добавления/редактирования вопроса."""

    def __init__(self, parent, question=None, next_id=None):
        super().__init__(parent)
        self.title("Вопрос")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        self._qid = question["id"] if question else next_id

        ttk.Label(self, text="ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Label(self, text=str(self._qid), font=("Arial", 10, "bold")).grid(
            row=0, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Label(self, text="Текст вопроса:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.text_var = tk.StringVar(value=question["text"] if question else "")
        ttk.Entry(self, textvariable=self.text_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ans_frame = ttk.LabelFrame(self, text="Варианты ответа")
        ans_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        existing = question.get("answers", list(_DEFAULT_ANSWERS)) if question else list(_DEFAULT_ANSWERS)
        self._answers = [dict(a) for a in existing]

        self._listbox = tk.Listbox(ans_frame, height=4, width=40)
        self._listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self._refresh_list()

        ans_btn = ttk.Frame(ans_frame)
        ans_btn.pack(side="left", padx=5)
        ttk.Button(ans_btn, text="Добавить", command=self._add_answer).pack(fill="x", pady=2)
        ttk.Button(ans_btn, text="Редактировать", command=self._edit_answer).pack(fill="x", pady=2)
        ttk.Button(ans_btn, text="Удалить", command=self._del_answer).pack(fill="x", pady=2)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.transient(parent)
        self.wait_window()

    def _refresh_list(self):
        self._listbox.delete(0, "end")
        for a in self._answers:
            self._listbox.insert("end", f"{a['text']}  (значение: {a['value']})")

    def _add_answer(self):
        dlg = AnswerChoiceDialog(self)
        if dlg.result:
            self._answers.append(dlg.result)
            self._refresh_list()

    def _edit_answer(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        dlg = AnswerChoiceDialog(self, choice=self._answers[idx])
        if dlg.result:
            self._answers[idx] = dlg.result
            self._refresh_list()

    def _del_answer(self):
        sel = self._listbox.curselection()
        if sel:
            del self._answers[sel[0]]
            self._refresh_list()

    def _ok(self):
        text = self.text_var.get().strip()
        if not text:
            messagebox.showerror("Ошибка", "Текст вопроса не может быть пустым.", parent=self)
            return
        if len(self._answers) < 2:
            messagebox.showerror("Ошибка", "Необходимо не менее двух вариантов ответа.", parent=self)
            return
        self.result = {"id": self._qid, "text": text, "answers": self._answers}
        self.destroy()


class OutcomeDialog(tk.Toplevel):
    """Диалог добавления/редактирования исхода с коэффициентами."""

    def __init__(self, parent, questions, outcome=None):
        super().__init__(parent)
        self.title("Исход")
        self.resizable(True, True)
        self.grab_set()
        self.result = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="Название:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.name_var = tk.StringVar(value=outcome["name"] if outcome else "")
        ttk.Entry(self, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Коэффициенты — прокручиваемый список
        coeff_outer = ttk.LabelFrame(self, text="Коэффициенты P(E=да | H) — от 0 до 1")
        coeff_outer.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        coeff_outer.columnconfigure(0, weight=1)
        coeff_outer.rowconfigure(0, weight=1)

        coeff_canvas = tk.Canvas(coeff_outer, highlightthickness=0)
        coeff_scroll = ttk.Scrollbar(coeff_outer, orient="vertical", command=coeff_canvas.yview)
        coeff_canvas.configure(yscrollcommand=coeff_scroll.set)
        coeff_scroll.pack(side="right", fill="y")
        coeff_canvas.pack(side="left", fill="both", expand=True)

        coeff_inner = ttk.Frame(coeff_canvas)
        win_id = coeff_canvas.create_window((0, 0), window=coeff_inner, anchor="nw")

        def _on_inner_configure(e):
            coeff_canvas.configure(scrollregion=coeff_canvas.bbox("all"))
        def _on_canvas_configure(e):
            coeff_canvas.itemconfig(win_id, width=e.width)
        coeff_inner.bind("<Configure>", _on_inner_configure)
        coeff_canvas.bind("<Configure>", _on_canvas_configure)

        self.coeff_vars = {}
        existing_coeffs = outcome.get("coefficients", {}) if outcome else {}
        for i, q in enumerate(questions):
            qid_str = str(q["id"])
            ttk.Label(
                coeff_inner,
                text=f"Q{q['id']}: {q['text']}",
                wraplength=260,
                justify="left",
            ).grid(row=i, column=0, padx=5, pady=2, sticky="w")
            var = tk.StringVar(value=str(existing_coeffs.get(qid_str, "0.5")))
            ttk.Entry(coeff_inner, textvariable=var, width=8).grid(row=i, column=1, padx=5, pady=2)
            self.coeff_vars[qid_str] = var

        # Ограничиваем высоту прокручиваемой области
        visible_rows = min(len(questions), 8)
        coeff_canvas.configure(height=visible_rows * 30 + 10)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.transient(parent)
        self.wait_window()

    def _ok(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Название не может быть пустым.", parent=self)
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

        self.result = {"name": name, "coefficients": coefficients}
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
        n = len(self.app.expert_system.outcomes)
        apriori = 1.0 / n if n > 0 else 0.0
        for o in self.app.expert_system.outcomes:
            self.o_tree.insert("", "end", values=(o["name"], f"{apriori:.2f}"))

    def _add_question(self):
        existing_ids = [q["id"] for q in self.app.expert_system.questions]
        next_id = max(existing_ids, default=0) + 1
        dlg = QuestionDialog(self, next_id=next_id)
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
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.start_btn = ttk.Button(left, text="Начать консультацию", command=self._start)
        self.start_btn.pack(pady=10)

        self.question_label = ttk.Label(left, text="", font=("Arial", 14), wraplength=400, justify="center")
        self.question_label.pack(pady=20)

        self.answers_frame = ttk.LabelFrame(left, text="Варианты ответа")
        self.answers_frame.pack(pady=5, fill="x", padx=10)

        self.selected_answer_idx = tk.IntVar(value=0)

        self.answer_btn = ttk.Button(left, text="Ответить", command=self._answer)
        self.answer_btn.pack(pady=10)
        self.answer_btn.pack_forget()

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
        self.answer_btn.pack(pady=10)
        self.start_btn.config(state="disabled")
        self._show_question()
        self._update_probabilities()

    def _show_question(self):
        es = self.app.expert_system
        q = es.questions[self.current_q_idx]
        self.question_label.config(
            text=f"Вопрос {self.current_q_idx + 1}/{len(es.questions)}:\n\n{q['text']}"
        )

        for widget in self.answers_frame.winfo_children():
            widget.destroy()

        choices = q.get("answers", list(_DEFAULT_ANSWERS))
        self.selected_answer_idx.set(0)
        for i, choice in enumerate(choices):
            rb = ttk.Radiobutton(
                self.answers_frame,
                text=choice["text"],
                variable=self.selected_answer_idx,
                value=i,
                command=self._on_answer_change,
            )
            rb.pack(anchor="w", padx=10, pady=2)

        self._on_answer_change()

    def _on_answer_change(self):
        es = self.app.expert_system
        if self.current_q_idx >= len(es.questions):
            return
        q = es.questions[self.current_q_idx]
        choices = q.get("answers", list(_DEFAULT_ANSWERS))
        idx = self.selected_answer_idx.get()
        if 0 <= idx < len(choices):
            self._update_probabilities(preview=(str(q["id"]), choices[idx]["value"]))

    def _update_probabilities(self, preview=None):
        es = self.app.expert_system
        answers = dict(self.answers)
        if preview is not None:
            qid_str, value = preview
            answers[qid_str] = value
        posteriors = es.compute_posteriors(answers)
        self.prob_tree.delete(*self.prob_tree.get_children())
        for name, prob in posteriors:
            self.prob_tree.insert("", "end", values=(name, f"{prob * 100:.1f}%"))

    def _answer(self):
        es = self.app.expert_system
        q = es.questions[self.current_q_idx]
        choices = q.get("answers", list(_DEFAULT_ANSWERS))
        idx = self.selected_answer_idx.get()
        value = choices[idx]["value"] if 0 <= idx < len(choices) else 0.5
        self.answers[str(q["id"])] = value
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

        for widget in self.answers_frame.winfo_children():
            widget.destroy()
        self.answer_btn.pack_forget()
        self.start_btn.config(state="normal")

        if posteriors:
            best_name, best_prob = posteriors[0]
            self.question_label.config(
                text=f"Наиболее вероятный исход:\n\n{best_name}\n(вероятность: {best_prob * 100:.1f}%)"
            )
            messagebox.showinfo(
                "Результат консультации",
                f"Наиболее вероятный исход: {best_name}\nВероятность: {best_prob * 100:.1f}%",
            )
        else:
            self.question_label.config(text="Нет данных для вывода.")


# ─── Вкладка «Дерево решений» ───────────────────────────────────────────────

class TreeTab(ttk.Frame):
    """Вкладка для визуализации дерева решений с панелью фильтрации."""

    NODE_W = 130
    NODE_H = 70
    H_GAP = 10
    V_GAP = 140

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.filter_vars = {}    # {qid_str: tk.IntVar}  -1=все варианты, >=0=индекс ответа
        self._known_q_ids = []
        self.show_prob_var = tk.BooleanVar(value=True)
        self._build_ui()

    def _build_ui(self):
        # ── Левая панель — фильтры ──
        self.filter_panel = ttk.LabelFrame(self, text="Фильтр по ответам")
        self.filter_panel.pack(side="left", fill="y", padx=(5, 0), pady=5)

        # Прокручиваемая область для фильтров
        filter_canvas = tk.Canvas(self.filter_panel, width=215, highlightthickness=0)
        filter_scrollbar = ttk.Scrollbar(self.filter_panel, orient="vertical", command=filter_canvas.yview)
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        filter_scrollbar.pack(side="right", fill="y")
        filter_canvas.pack(side="left", fill="both", expand=True)

        self.filter_inner = ttk.Frame(filter_canvas)
        self._filter_win_id = filter_canvas.create_window((0, 0), window=self.filter_inner, anchor="nw")

        self.filter_inner.bind(
            "<Configure>",
            lambda e: filter_canvas.configure(scrollregion=filter_canvas.bbox("all")),
        )
        filter_canvas.bind(
            "<Configure>",
            lambda e: filter_canvas.itemconfig(self._filter_win_id, width=e.width),
        )
        self._filter_canvas = filter_canvas

        # ── Правая часть — тулбар + холст дерева ──
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x", padx=5, pady=5)
        ttk.Button(toolbar, text="Построить дерево", command=self._draw_tree).pack(side="left")
        ttk.Button(toolbar, text="Сохранить изображение", command=self._save_image).pack(side="left", padx=5)
        ttk.Checkbutton(
            toolbar,
            text="Показывать вероятности",
            variable=self.show_prob_var,
            command=self._draw_tree,
        ).pack(side="left", padx=10)

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    def _build_filter_panel(self):
        """Перестраивает элементы фильтрации по текущим вопросам системы."""
        for widget in self.filter_inner.winfo_children():
            widget.destroy()
        self.filter_vars = {}     # {qid_str: tk.StringVar}
        self.filter_choices = {}  # {qid_str: [answers]}

        es = self.app.expert_system
        for q in es.questions:
            qid_str = str(q["id"])
            answers = q.get("answers", [])
            self.filter_choices[qid_str] = answers

            ttk.Label(
                self.filter_inner,
                text=f"Q{q['id']}: {q['text']}",
                wraplength=195,
                font=("Arial", 9, "bold"),
                justify="left",
            ).pack(anchor="w", padx=4, pady=(10, 2))

            _ALL = "— Все варианты —"
            values = [_ALL] + [a["text"] for a in answers]
            var = tk.StringVar(value=_ALL)
            self.filter_vars[qid_str] = var

            cb = ttk.Combobox(
                self.filter_inner,
                textvariable=var,
                values=values,
                state="readonly",
                width=26,
            )
            cb.pack(anchor="w", padx=4, pady=2)
            cb.bind("<<ComboboxSelected>>", lambda e: self._draw_tree())

            ttk.Separator(self.filter_inner, orient="horizontal").pack(fill="x", padx=4, pady=6)

    def _get_active_filters(self):
        """Возвращает {qid_str: answer_index} для активных (не «все») фильтров."""
        _ALL = "— Все варианты —"
        result = {}
        for qid, var in self.filter_vars.items():
            sel = var.get()
            if sel == _ALL:
                continue
            choices = self.filter_choices.get(qid, [])
            for i, a in enumerate(choices):
                if a["text"] == sel:
                    result[qid] = i
                    break
        return result

    def _calc_effective_leaves(self, es, active_filters):
        """Число листьев дерева с учётом активных фильтров."""
        n = 1
        for q in es.questions:
            if str(q["id"]) not in active_filters:
                n *= len(q.get("answers", _DEFAULT_ANSWERS))
        return max(n, 1)

    def _outlined_text(self, x, y, text, fill="white", outline_color="black", **kwargs):
        """Рисует текст с однопиксельной обводкой."""
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.canvas.create_text(x + dx, y + dy, text=text, fill=outline_color, **kwargs)
        self.canvas.create_text(x, y, text=text, fill=fill, **kwargs)

    def _save_image(self):
        if not self.canvas.find_all():
            messagebox.showwarning("Внимание", "Сначала постройте дерево.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG изображение", "*.png"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            import textwrap
            from PIL import Image, ImageDraw, ImageFont

            es = self.app.expert_system
            if not es.questions or not es.outcomes:
                return

            active_filters = self._get_active_filters()
            show_prob = self.show_prob_var.get()

            SCALE = 2
            NW = self.NODE_W * SCALE
            NH = self.NODE_H * SCALE
            HGAP = self.H_GAP * SCALE
            VGAP = self.V_GAP * SCALE
            FONT_SIZE = 16 * SCALE

            n_leaves = self._calc_effective_leaves(es, active_filters)
            n_q = len(es.questions)
            tree_w = int(max(n_leaves * (NW + HGAP), NW + HGAP))
            tree_h = int((n_q + 1) * (NH + VGAP) + 80 * SCALE)

            font = None
            for fp in [
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ]:
                try:
                    font = ImageFont.truetype(fp, FONT_SIZE)
                    break
                except Exception:
                    pass
            if font is None:
                font = ImageFont.load_default()

            img = Image.new("RGB", (tree_w, tree_h), "white")
            draw = ImageDraw.Draw(img)

            def pil_text(x, y, text, fill="white", outline="#000000"):
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.multiline_text(
                        (x + dx, y + dy), text, fill=outline,
                        font=font, anchor="mm", align="center",
                    )
                draw.multiline_text(
                    (x, y), text, fill=fill,
                    font=font, anchor="mm", align="center",
                )

            def draw_node(x, y, q_idx, answers, half_width):
                x, y, half_width = int(x), int(y), int(half_width)
                if q_idx >= len(es.questions):
                    posteriors = es.compute_posteriors(answers)
                    if posteriors:
                        name = "\n".join(textwrap.wrap(posteriors[0][0], 18))
                        if show_prob:
                            text = f"{name}\n{posteriors[0][1] * 100:.1f}%"
                        else:
                            text = name
                    else:
                        text = "?"
                    draw.rectangle(
                        [x - NW // 2, y, x + NW // 2, y + NH],
                        fill="#90EE90", outline="#2E8B57", width=2 * SCALE,
                    )
                    pil_text(x, y + NH // 2, text)
                    return

                q = es.questions[q_idx]
                qid_str = str(q["id"])
                filter_idx = active_filters.get(qid_str, None)

                label = "\n".join(textwrap.wrap(f"Q{q['id']}: {q['text']}", 20))
                draw.rectangle(
                    [x - NW // 2, y, x + NW // 2, y + NH],
                    fill="#87CEEB", outline="#4682B4", width=2 * SCALE,
                )
                pil_text(x, y + NH // 2, label)

                choices = q.get("answers", list(_DEFAULT_ANSWERS))
                if filter_idx is not None and 0 <= filter_idx < len(choices):
                    displayed = [(filter_idx, choices[filter_idx])]
                    n_eff = 1
                else:
                    displayed = list(enumerate(choices))
                    n_eff = len(choices)

                child_y = y + NH + VGAP
                child_half = half_width / n_eff
                for slot, (orig_idx, choice) in enumerate(displayed):
                    child_x = x - half_width + child_half * (2 * slot + 1)
                    child_answers = dict(answers)
                    child_answers[qid_str] = choice.get("value", 0.5)
                    is_sel = filter_idx is not None and orig_idx == filter_idx
                    line_color = "#1565C0" if is_sel else "#555555"
                    line_w = 3 * SCALE if is_sel else SCALE
                    draw.line(
                        [(x, y + NH), (int(child_x), int(child_y))],
                        fill=line_color, width=line_w,
                    )
                    mid_x = int((x + child_x) / 2)
                    mid_y = int((y + NH + child_y) / 2)
                    if is_sel:
                        pil_text(mid_x, mid_y, choice["text"], fill="#FFD600", outline="#1565C0")
                    else:
                        pil_text(mid_x, mid_y, choice["text"])
                    draw_node(child_x, child_y, q_idx + 1, child_answers, child_half)

            draw_node(tree_w / 2, 30 * SCALE, 0, {}, tree_w / 2)
            img.save(path, "PNG")
            messagebox.showinfo("Сохранено", f"Дерево сохранено в:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")

    def _draw_tree(self):
        es = self.app.expert_system
        if not es.questions or not es.outcomes:
            messagebox.showwarning("Внимание", "Система должна содержать вопросы и исходы.")
            return

        # Перестроить панель фильтров, если состав вопросов изменился
        current_q_ids = [q["id"] for q in es.questions]
        if current_q_ids != self._known_q_ids:
            self._build_filter_panel()
            self._known_q_ids = list(current_q_ids)

        active_filters = self._get_active_filters()
        effective_leaves = self._calc_effective_leaves(es, active_filters)

        self.canvas.delete("all")

        n_q = len(es.questions)
        tree_w = max(effective_leaves * (self.NODE_W + self.H_GAP), self.NODE_W + self.H_GAP)
        tree_h = (n_q + 1) * (self.NODE_H + self.V_GAP) + 40

        # Центрируем дерево: если оно уже холста — отступ по центру
        canvas_w = self.canvas.winfo_width() or 600
        margin = max((canvas_w - tree_w) / 2, 0)
        root_x = margin + tree_w / 2
        full_w = margin + tree_w + margin

        self._draw_node(es, root_x, 30, 0, {}, tree_w / 2, active_filters)
        self.canvas.configure(scrollregion=(0, 0, full_w, tree_h))

    def _draw_node(self, es, x, y, q_idx, answers, half_width, active_filters):
        """Рекурсивно рисует узлы дерева с учётом активных фильтров."""
        if q_idx >= len(es.questions):
            posteriors = es.compute_posteriors(answers)
            if posteriors:
                best_name, best_prob = posteriors[0]
                # Обрезаем длинное название до 2 строк по ~20 символов
                words = best_name.split()
                lines, cur = [], ""
                for w in words:
                    if len(cur) + len(w) + 1 > 20:
                        lines.append(cur.rstrip())
                        cur = w + " "
                        if len(lines) == 2:
                            break
                    else:
                        cur += w + " "
                else:
                    lines.append(cur.rstrip())
                short_name = "\n".join(lines[:2])
                if len(lines) > 2 or len(" ".join(words)) > len(" ".join(lines[:2].copy())):
                    short_name = short_name.rstrip() + "…"
                if self.show_prob_var.get():
                    text = f"{short_name}\n{best_prob * 100:.1f}%"
                else:
                    text = short_name
            else:
                text = "?"
            self.canvas.create_rectangle(
                x - self.NODE_W // 2, y, x + self.NODE_W // 2, y + self.NODE_H,
                fill="#90EE90", outline="#2E8B57", width=2,
            )
            self._outlined_text(x, y + self.NODE_H // 2, text,
                                 font=("Arial", 11), width=self.NODE_W - 10, justify="center")
            return

        q = es.questions[q_idx]
        qid_str = str(q["id"])
        filter_idx = active_filters.get(qid_str, None)

        # Обрезаем текст вопроса до 2 строк по ~20 символов
        q_words = q["text"].split()
        q_lines, q_cur = [], ""
        for w in q_words:
            if len(q_cur) + len(w) + 1 > 20:
                q_lines.append(q_cur.rstrip())
                q_cur = w + " "
                if len(q_lines) == 2:
                    break
            else:
                q_cur += w + " "
        else:
            q_lines.append(q_cur.rstrip())
        q_short = "\n".join(q_lines[:2])
        if len(q_lines) > 2 or len(q["text"]) > len(" ".join(q_lines[:2])):
            q_short = q_short.rstrip() + "…"
        node_label = f"Q{q['id']}: {q_short}"

        self.canvas.create_rectangle(
            x - self.NODE_W // 2, y, x + self.NODE_W // 2, y + self.NODE_H,
            fill="#87CEEB", outline="#4682B4", width=2,
        )
        self._outlined_text(x, y + self.NODE_H // 2,
                             node_label,
                             font=("Arial", 11), width=self.NODE_W - 10, justify="center")

        choices = q.get("answers", list(_DEFAULT_ANSWERS))
        if filter_idx is not None and 0 <= filter_idx < len(choices):
            displayed = [(filter_idx, choices[filter_idx])]
            n_eff = 1
        else:
            displayed = list(enumerate(choices))
            n_eff = len(choices)

        child_y = y + self.NODE_H + self.V_GAP
        child_half = half_width / n_eff

        for slot, (orig_idx, choice) in enumerate(displayed):
            child_x = x - half_width + child_half * (2 * slot + 1)
            child_answers = dict(answers)
            child_answers[qid_str] = choice.get("value", 0.5)

            is_sel = filter_idx is not None and orig_idx == filter_idx
            if is_sel:
                self.canvas.create_line(
                    x, y + self.NODE_H, child_x, child_y,
                    fill="#1565C0", width=3,
                )
            else:
                self.canvas.create_line(x, y + self.NODE_H, child_x, child_y, fill="#555")

            mid_x = (x + child_x) / 2
            mid_y = (y + self.NODE_H + child_y) / 2
            label = choice["text"]
            if len(label) > 15:
                label = label[:14] + "\u2026"
            if is_sel:
                self._outlined_text(
                    mid_x, mid_y, label,
                    fill="#FFD600", outline_color="#1565C0",
                    font=("Arial", 8, "bold"),
                )
            else:
                self._outlined_text(mid_x, mid_y, label, font=("Arial", 8, "bold"))

            self._draw_node(es, child_x, child_y, q_idx + 1, child_answers, child_half, active_filters)


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
