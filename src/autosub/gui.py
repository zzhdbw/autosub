"""AutoSub GUI — cross-platform desktop application."""

import re
import sys
import threading
import queue
from pathlib import Path

import customtkinter as ctk
from loguru import logger

from autosub.main import main as run_cli
from autosub.model_manager import MODELS, status as model_status, download as download_model

# ── Appearance ──────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT = ("Segoe UI", 13)
FONT_MONO = ("Cascadia Code", 11) if sys.platform == "win32" else ("Menlo", 11)

# ── Log capture ─────────────────────────────────────────────────────────


class LogCapture:
    """Capture loguru output (via stderr) for real-time GUI display."""

    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()

    def write(self, text: str) -> None:
        if text.strip():
            self.queue.put(text)

    def flush(self) -> None:
        pass


# ── Progress helpers ───────────────────────────────────────────────────

_STEP_RE = re.compile(r"Step (\d+)/4:")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

LEVEL_COLORS = {
    "TRACE": ("gray", "gray"),
    "DEBUG": ("gray", "gray"),
    "INFO": ("white", "white"),
    "WARNING": ("orange", "orange"),
    "ERROR": ("#E06C75", "#E06C75"),
    "CRITICAL": ("#E06C75", "#E06C75"),
}


def _tag_from_log(text: str) -> str | None:
    """Return the loguru level tag (e.g. ``INFO``) if present."""
    for level in LEVEL_COLORS:
        if f"| {level:<7} |" in text:
            return level
    return None


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ── Main application ───────────────────────────────────────────────────


class App(ctk.CTk):
    """AutoSub GUI."""

    STAGE_LABELS = [
        "Audio Extraction",
        "Voice Activity Detection",
        "Speech Recognition (ASR)",
        "Translation",
        "Generate Subtitles",
    ]

    def __init__(self) -> None:
        super().__init__()

        self.title("AutoSub — Auto Subtitle Generator")
        self.geometry("880x820")
        self.minsize(720, 640)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── state ────────────────────────────────────────────────────
        self.running = False
        self._worker: threading.Thread | None = None
        self._capture: LogCapture | None = None
        self._old_stderr: object | None = None
        self._output_srt: str | None = None

        # ── ui vars ──────────────────────────────────────────────────
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar(value="output")
        self.backend = ctk.StringVar(value="llamacpp")
        self.skip_vad = ctk.BooleanVar(value=False)
        self.skip_asr = ctk.BooleanVar(value=False)
        self.skip_translate = ctk.BooleanVar(value=False)

        self._build_ui()
        self._after_id: str | None = None

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(7, weight=1)  # log area expands

        # ── Title ────────────────────────────────────────────────────
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, pady=(16, 8), padx=20, sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            title_frame, text="AutoSub — Auto Subtitle Generator",
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self._models_btn = ctk.CTkButton(
            title_frame, text="\U0001F4E6 Models", font=FONT,
            width=100, command=self._open_model_manager,
        )
        self._models_btn.grid(row=0, column=1, sticky="e")

        # ── Input file ───────────────────────────────────────────────
        self._add_file_row(1, "Input Video", self.input_path, self._browse_input,
                           filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                                      ("All files", "*.*")])

        # ── Output dir ───────────────────────────────────────────────
        self._add_dir_row(2, "Output Dir", self.output_dir, self._browse_output)

        # ── Separator ────────────────────────────────────────────────
        ctk.CTkFrame(self, height=1).grid(row=3, column=0, sticky="ew", padx=20, pady=4)

        # ── Backend selection ────────────────────────────────────────
        backend_frame = ctk.CTkFrame(self)
        backend_frame.grid(row=4, column=0, pady=8, padx=20, sticky="ew")
        backend_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(backend_frame, text="Backend:", font=FONT).grid(
            row=0, column=0, padx=(0, 12), sticky="w")
        self._backend_menu = ctk.CTkOptionMenu(
            backend_frame, values=["llamacpp", "transformers"],
            variable=self.backend, command=self._on_backend_change,
        )
        self._backend_menu.grid(row=0, column=1, sticky="w", padx=(0, 12))

        self._backend_info = ctk.CTkLabel(
            backend_frame, text="", font=("Segoe UI", 11),
        )
        self._backend_info.grid(row=0, column=2, sticky="w")
        self._on_backend_change(self.backend.get())

        # ── Skip checkboxes ──────────────────────────────────────────
        skip_frame = ctk.CTkFrame(self)
        skip_frame.grid(row=5, column=0, pady=4, padx=20, sticky="ew")
        for i, (var, label) in enumerate([
            (self.skip_vad, "Skip VAD"),
            (self.skip_asr, "Skip ASR"),
            (self.skip_translate, "Skip Translate"),
        ]):
            cb = ctk.CTkCheckBox(skip_frame, text=label, variable=var, font=FONT)
            cb.grid(row=0, column=i, padx=(0, 20), pady=4, sticky="w")

        # ── Start button ─────────────────────────────────────────────
        self._start_btn = ctk.CTkButton(
            self, text="\u25b6  Start Processing",
            font=("Segoe UI", 14, "bold"),
            height=40, command=self._toggle_pipeline,
        )
        self._start_btn.grid(row=6, column=0, pady=(10, 4), padx=20, sticky="ew")

        # ── Stage progress ───────────────────────────────────────────
        self._stage_labels: list[ctk.CTkLabel] = []
        stage_frame = ctk.CTkFrame(self)
        stage_frame.grid(row=7, column=0, pady=4, padx=20, sticky="ew")
        stage_frame.grid_columnconfigure(tuple(range(5)), weight=1)

        for idx, label in enumerate(self.STAGE_LABELS):
            f = ctk.CTkFrame(stage_frame, corner_radius=6)
            f.grid(row=0, column=idx, padx=3, pady=6, sticky="ew")
            f.grid_columnconfigure(0, weight=1)
            lbl = ctk.CTkLabel(
                f, text=f"{idx+1}. {label}",
                font=("Segoe UI", 11), anchor="center",
            )
            lbl.grid(row=0, column=0, padx=4, pady=3, sticky="ew")
            self._stage_labels.append(lbl)

        # ── Log output ───────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=8, column=0, pady=(6, 2), padx=20, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self._log_text = ctk.CTkTextbox(
            log_frame, font=FONT_MONO, wrap="word", state="disabled",
        )
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # ── Bottom bar ───────────────────────────────────────────────
        bar = ctk.CTkFrame(self)
        bar.grid(row=9, column=0, pady=(2, 12), padx=20, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(bar, text="Ready", font=FONT, anchor="w")
        self._status_label.grid(row=0, column=0, sticky="ew")

        self._open_srt_btn = ctk.CTkButton(
            bar, text="Open SRT", font=FONT,
            state="disabled", command=self._open_srt,
        )
        self._open_srt_btn.grid(row=0, column=1, padx=(4, 0))

        self._open_dir_btn = ctk.CTkButton(
            bar, text="Open Output Dir", font=FONT,
            state="disabled", command=self._open_output_dir,
        )
        self._open_dir_btn.grid(row=0, column=2, padx=(4, 0))

    # ── Helper: file row ─────────────────────────────────────────────

    def _add_file_row(self, row: int, label: str, var: ctk.StringVar,
                      command, filetypes) -> None:
        f = ctk.CTkFrame(self)
        f.grid(row=row, column=0, pady=4, padx=20, sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text=f"{label}:", font=FONT, width=90, anchor="w").grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        ctk.CTkEntry(f, textvariable=var, font=FONT).grid(
            row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(f, text="Browse", font=FONT, width=80, command=command).grid(
            row=0, column=2)

    def _add_dir_row(self, row: int, label: str, var: ctk.StringVar,
                     command) -> None:
        f = ctk.CTkFrame(self)
        f.grid(row=row, column=0, pady=4, padx=20, sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text=f"{label}:", font=FONT, width=90, anchor="w").grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        ctk.CTkEntry(f, textvariable=var, font=FONT).grid(
            row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(f, text="Browse", font=FONT, width=80, command=command).grid(
            row=0, column=2)

    # ── Browse callbacks ─────────────────────────────────────────────

    def _browse_input(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                       ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)

    def _browse_output(self) -> None:
        from tkinter import filedialog
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def _on_backend_change(self, value: str) -> None:
        self._backend_info.configure(
            text="(GGUF, ~1.1 GB, recommended)" if value == "llamacpp"
            else "(Transformers, ~3.8 GB, slower)"
        )

    # ── Model management ────────────────────────────────────────────

    def _open_model_manager(self) -> None:
        ModelManagerDialog(self)

    # ── Pipeline control ─────────────────────────────────────────────

    def _toggle_pipeline(self) -> None:
        if self.running:
            return
        if not self.input_path.get():
            self._log("Select an input video file first.\n")
            return
        self.running = True
        self._start_btn.configure(state="disabled", text="\u23f3 Processing\u2026")

        # Reset stages
        for lbl in self._stage_labels:
            lbl.master.configure(fg_color=("gray75", "gray25"))
            lbl.configure(text_color=("gray30", "gray60"))
        self._output_srt = None
        self._open_srt_btn.configure(state="disabled")
        self._open_dir_btn.configure(state="disabled")

        # Redirect stderr (loguru output) to GUI
        self._capture = LogCapture()
        self._old_stderr = sys.stderr
        sys.stderr = self._capture  # type: ignore[assignment]

        # Run in thread
        self._worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self._worker.start()

        # Poll log queue
        if self._after_id is None:
            self._poll_log()

    def _run_pipeline(self) -> None:
        try:
            sys.argv = [
                "autosub",
                self.input_path.get(),
                "--output-dir", self.output_dir.get(),
                "--backend", self.backend.get(),
            ]
            if self.backend.get() == "llamacpp":
                sys.argv[1:1] = []  # gguf-model has a default, no need to force
            if self.skip_vad.get():
                sys.argv.append("--skip-vad")
            if self.skip_asr.get():
                sys.argv.append("--skip-asr")
            if self.skip_translate.get():
                sys.argv.append("--skip-translate")

            run_cli()
        except SystemExit:
            pass
        except Exception as exc:
            logger.error("Pipeline error: {}", exc)
        finally:
            self.running = False
            if self._after_id:
                self.after_cancel(self._after_id)
                self._after_id = None
            self.after(0, self._on_complete)

    def _on_complete(self) -> None:
        # Restore stderr
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr  # type: ignore[assignment]
            self._old_stderr = None

        self._start_btn.configure(state="normal", text="\u25b6  Start Processing")
        self._status_label.configure(text="Done.")

        # Try to find output SRT
        if self._output_srt:
            self._open_srt_btn.configure(state="normal")
        self._open_dir_btn.configure(state="normal")

    # ── Log polling ──────────────────────────────────────────────────

    def _poll_log(self) -> None:
        if self._capture is None:
            self._after_id = self.after(100, self._poll_log)
            return

        # Drain queue
        texts: list[str] = []
        while not self._capture.queue.empty():
            texts.append(self._capture.queue.get_nowait())

        if texts:
            block = "".join(texts)
            self._append_log(block)
            self._update_stages(block)

        if self.running:
            self._after_id = self.after(100, self._poll_log)
        else:
            self._after_id = None

    def _append_log(self, text: str) -> None:
        self._log_text.configure(state="normal")
        text = _strip_ansi(text)

        for line in text.splitlines(keepends=True):
            tag = _tag_from_log(line)
            if tag == "ERROR" or tag == "CRITICAL":
                c = "#E06C75"
            elif tag == "WARNING":
                c = "#E5C07B"
            else:
                c = None
            if c:
                self._log_text.insert("end", line, ("color",))
                self._log_text.tag_config("color", foreground=c)
            else:
                self._log_text.insert("end", line)

        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ── Stage progress ───────────────────────────────────────────────

    def _update_stages(self, text: str) -> None:
        m = _STEP_RE.search(text)
        if m:
            step = int(m.group(1)) - 1
            for idx, lbl in enumerate(self._stage_labels):
                if idx < step:
                    lbl.master.configure(fg_color=("#2E7D32", "#388E3C"))
                    lbl.configure(text_color=("white", "#A5D6A7"))
                elif idx == step:
                    lbl.master.configure(fg_color=("#1565C0", "#1976D2"))
                    lbl.configure(text_color=("white", "#90CAF9"))

        if "Done." in text or "\u2713 Subtitles" in text:
            for idx, lbl in enumerate(self._stage_labels):
                lbl.master.configure(fg_color=("#2E7D32", "#388E3C"))
                lbl.configure(text_color=("white", "#A5D6A7"))
            self._status_label.configure(text="Completed successfully!")

        m2 = re.search(r"\u2713 Subtitles \u2192 (.+)", text)
        if m2:
            self._output_srt = m2.group(1).strip()

    # ── Window close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self.running:
            from tkinter.messagebox import askyesno
            if not askyesno("Quit", "Processing is in progress. Quit anyway?"):
                return
        self.destroy()

    # ── Output actions ───────────────────────────────────────────────

    def _open_srt(self) -> None:
        if self._output_srt and Path(self._output_srt).exists():
            if sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", self._output_srt], check=False)
            elif sys.platform == "win32":
                import subprocess
                subprocess.run(["start", self._output_srt], shell=True, check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", self._output_srt], check=False)

    def _open_output_dir(self) -> None:
        d = Path(self.output_dir.get())
        if d.exists():
            if sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(d)], check=False)
            elif sys.platform == "win32":
                import subprocess
                subprocess.run(["explorer", str(d)], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", str(d)], check=False)

    def _log(self, text: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")


# ── Model Manager Dialog ────────────────────────────────────────────────


class ModelManagerDialog(ctk.CTkToplevel):
    """Modal dialog for downloading models."""

    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        self.parent = parent

        self.title("Models Management")
        self.geometry("720x560")
        self.minsize(600, 400)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=0)

        # ── Header ─────────────────────────────────────────────────
        header = ctk.CTkLabel(
            self, text="Download Required Models",
            font=("Segoe UI", 15, "bold"),
        )
        header.grid(row=0, column=0, pady=(14, 6), padx=20, sticky="w")

        # ── Model list ─────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(self)
        scroll.grid(row=1, column=0, pady=4, padx=20, sticky="nsew")
        scroll.grid_columnconfigure(1, weight=1)

        self._model_rows: dict[str, dict] = {}
        for idx, key in enumerate(MODELS):
            info = MODELS[key]
            row_frame = ctk.CTkFrame(scroll)
            row_frame.grid(row=idx, column=0, pady=3, sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)

            name_lbl = ctk.CTkLabel(
                row_frame, text=info["name"],
                font=("Segoe UI", 13, "bold"), anchor="w",
            )
            name_lbl.grid(row=0, column=0, padx=(10, 6), pady=6, sticky="w")

            desc_lbl = ctk.CTkLabel(
                row_frame, text=info["description"],
                font=("Segoe UI", 11), anchor="w",
            )
            desc_lbl.grid(row=0, column=1, padx=6, pady=6, sticky="w")

            status_lbl = ctk.CTkLabel(
                row_frame, text="", font=("Segoe UI", 11, "bold"), anchor="e",
            )
            status_lbl.grid(row=0, column=2, padx=6, pady=6, sticky="e")

            dl_btn = ctk.CTkButton(
                row_frame, text="Download", font=FONT,
                width=90, height=28,
                command=lambda k=key: self._download_one(k),
            )
            dl_btn.grid(row=0, column=3, padx=(6, 10), pady=6)

            self._model_rows[key] = {
                "frame": row_frame,
                "status_lbl": status_lbl,
                "dl_btn": dl_btn,
            }

        # ── Bottom bar ─────────────────────────────────────────────
        bar = ctk.CTkFrame(self)
        bar.grid(row=2, column=0, pady=(6, 0), padx=20, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self._dl_all_btn = ctk.CTkButton(
            bar, text="Download All", font=FONT,
            command=self._download_all,
        )
        self._dl_all_btn.grid(row=0, column=0, padx=(0, 12), pady=6, sticky="w")

        self._prog_bar = ctk.CTkProgressBar(bar)
        self._prog_bar.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self._prog_bar.set(0)

        self._prog_label = ctk.CTkLabel(
            bar, text="", font=("Segoe UI", 11), anchor="e",
        )
        self._prog_label.grid(row=0, column=2, padx=6, pady=6, sticky="e")

        close_btn = ctk.CTkButton(
            bar, text="Close", font=FONT, command=self.destroy,
        )
        close_btn.grid(row=0, column=3, padx=(6, 0), pady=6)

        # ── Log output ─────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, pady=(4, 12), padx=20, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self._log_text = ctk.CTkTextbox(
            log_frame, font=FONT_MONO, wrap="word", state="disabled",
            height=120,
        )
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # ── State ──────────────────────────────────────────────────
        self._downloading = False
        self._refresh_status()

    def _refresh_status(self) -> None:
        for key, row in self._model_rows.items():
            st = model_status(key)
            if st == "ok":
                row["status_lbl"].configure(
                    text="\u2713 Downloaded", text_color="#4CAF50",
                )
                row["dl_btn"].configure(state="disabled", text="\u2713")
            elif st == "partial":
                row["status_lbl"].configure(
                    text="Partial", text_color="#FF9800",
                )
                row["dl_btn"].configure(state="normal", text="Resume")
            else:
                row["status_lbl"].configure(
                    text="Not Downloaded", text_color="#757575",
                )
                row["dl_btn"].configure(state="normal", text="Download")

    def _download_one(self, key: str) -> None:
        if self._downloading:
            return
        self._downloading = True
        self._set_buttons_disabled(True)
        self._prog_bar.set(0)
        self._prog_label.configure(text="")
        self._clear_log()

        def _run() -> None:
            try:
                download_model(key, self._append_log, self._set_progress)
                self.after(0, self._on_dl_done, key, None)
            except Exception as exc:
                self.after(0, self._on_dl_done, key, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _download_all(self) -> None:
        if self._downloading:
            return
        self._downloading = True
        self._set_buttons_disabled(True)
        self._prog_bar.set(0)
        self._prog_label.configure(text="")
        self._clear_log()

        def _chain() -> None:
            keys = list(MODELS)
            failed = None
            for key in keys:
                self.after(0, lambda k=key: self._prog_label.configure(
                    text=f"Downloading {MODELS[k]['name']}..."
                ))
                try:
                    download_model(key, self._append_log, self._set_progress)
                except Exception as exc:
                    failed = str(exc)
                    break
            self.after(0, self._on_dl_all_done, failed)

        threading.Thread(target=_chain, daemon=True).start()

    def _on_dl_done(self, key: str, error: str | None) -> None:
        self._downloading = False
        self._refresh_status()
        self._set_buttons_disabled(False)
        if error:
            self._append_log(f"FAILED: {error}\n")
            self._prog_label.configure(text="Failed")
        else:
            self._prog_label.configure(text="Complete!")

    def _on_dl_all_done(self, error: str | None) -> None:
        self._downloading = False
        self._refresh_status()
        self._set_buttons_disabled(False)
        if error:
            self._append_log(f"FAILED: {error}\n")
            self._prog_label.configure(text="Failed")
        else:
            self._prog_label.configure(text="All Complete!")

    def _set_progress(self, pct: float) -> None:
        self.after(0, lambda: self._prog_bar.set(pct / 100))
        self.after(0, lambda: self._prog_label.configure(
            text=f"{pct:.0f}%"
        ))

    def _set_buttons_disabled(self, disabled: bool) -> None:
        st = "disabled" if disabled else "normal"
        for row in self._model_rows.values():
            row["dl_btn"].configure(state=st)
        self._dl_all_btn.configure(state=st)

    def _clear_log(self) -> None:
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        """Thread-safe log append (uses ``after`` to run on main thread)."""
        self.after(0, self._do_append_log, text)

    def _do_append_log(self, text: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")


# ── Entry point ────────────────────────────────────────────────────────


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
