from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageOps, ImageTk
except Exception:
    Image = None
    ImageOps = None
    ImageTk = None


def disable_mouse_wheel(widget: tk.Widget) -> None:
    widget.bind("<MouseWheel>", lambda _event: "break")
    widget.bind("<Button-4>", lambda _event: "break")
    widget.bind("<Button-5>", lambda _event: "break")


def make_labeled_combo(
    parent: ttk.Frame,
    row: int,
    column: int,
    label: str,
    variable: tk.StringVar,
    values: Sequence[str],
    *,
    width: int = 16,
    padx: tuple[int, int] = (8, 12),
    pady: tuple[int, int] = (8, 0),
    on_change: Callable[[object | None], None] | None = None,
) -> ttk.Combobox:
    label_widget = ttk.Label(parent, text=label, style="Field.TLabel")
    label_widget.grid(row=row, column=column, sticky="w", pady=pady)
    combo = ttk.Combobox(
        parent,
        textvariable=variable,
        state="readonly",
        values=list(values),
        width=width,
    )
    combo.grid(row=row, column=column + 1, sticky="ew", padx=padx, pady=pady)
    if on_change is not None:
        combo.bind("<<ComboboxSelected>>", on_change)
    combo._label_widget = label_widget  # type: ignore[attr-defined]
    disable_mouse_wheel(combo)
    return combo


def make_labeled_entry(
    parent: ttk.Frame,
    row: int,
    column: int,
    label: str,
    variable: tk.StringVar,
    *,
    width: int = 16,
    padx: tuple[int, int] = (8, 12),
    pady: tuple[int, int] = (8, 0),
) -> ttk.Entry:
    ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=column, sticky="w", pady=pady)
    entry = ttk.Entry(parent, textvariable=variable, width=width)
    entry.grid(row=row, column=column + 1, sticky="ew", padx=padx, pady=pady)
    disable_mouse_wheel(entry)
    return entry


class BlockCard(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Misc,
        title: str,
        summary: str = "",
        *,
        accent: str = "#5bb6ff",
        wraplength: int = 220,
    ) -> None:
        super().__init__(parent, style="Card.TFrame", padding=0)

        strip = tk.Frame(self, bg=accent, width=5, height=1, bd=0, highlightthickness=0)
        strip.pack(side="left", fill="y")

        body = ttk.Frame(self, style="Card.TFrame", padding=12)
        body.pack(side="left", fill="both", expand=True)

        self._title_var = tk.StringVar(value=title)
        self._summary_var = tk.StringVar(value=summary)

        ttk.Label(body, textvariable=self._title_var, style="BlockTitle.TLabel").pack(anchor="w")
        ttk.Label(
            body,
            textvariable=self._summary_var,
            style="BlockText.TLabel",
            wraplength=wraplength,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

    def set_title(self, title: str) -> None:
        self._title_var.set(title)

    def set_summary(self, summary: str) -> None:
        self._summary_var.set(summary)


class ImagePreview(ttk.Frame):
    def __init__(self, parent: ttk.Misc, *, fallback_text: str = "No image selected") -> None:
        super().__init__(parent, style="Card.TFrame", padding=0)
        self._fallback_text = fallback_text
        self._image_label = tk.Label(
            self,
            bg="#0b1522",
            fg="#eef5ff",
            text=fallback_text,
            font=("Segoe UI", 11),
            bd=0,
            highlightthickness=0,
        )
        self._image_label.pack(fill="both", expand=True, padx=12, pady=12)
        self._photo_image = None
        self._current_path: Path | None = None

    def clear(self, text: str | None = None) -> None:
        self._current_path = None
        self._photo_image = None
        self._image_label.configure(image="", text=text or self._fallback_text)

    def show(self, path: Path) -> None:
        self._current_path = path
        if Image is None or ImageTk is None or ImageOps is None or not path.exists():
            self.clear(path.name if path.exists() else self._fallback_text)
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            self.clear(path.name)
            return

        width = max(self._image_label.winfo_width() - 24, 640)
        height = max(self._image_label.winfo_height() - 24, 420)
        image = ImageOps.contain(image, (width, height))
        self._photo_image = ImageTk.PhotoImage(image)
        self._image_label.configure(image=self._photo_image, text="")


class LogConsole(ttk.Frame):
    def __init__(self, parent: ttk.Misc) -> None:
        super().__init__(parent, style="Card.TFrame", padding=0)
        box = ttk.Frame(self, style="Card.TFrame")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)

        self._text = tk.Text(
            box,
            bg="#0b1522",
            fg="#edf4ff",
            insertbackground="#edf4ff",
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(box, orient="vertical", command=self._text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scrollbar.set)

    def append(self, line: str) -> None:
        self._text.insert("end", line.rstrip() + "\n")
        self._text.see("end")

    def clear(self) -> None:
        self._text.delete("1.0", "end")
