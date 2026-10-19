"""Tkinter front end: pick a .vrm, pick an armor set, press Install."""

import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from . import game as G
from .build import DEFAULT_ARMOR, KNOWN_ARMOR, Options, armor_sets, build
from .chunk import GameFiles

LOADER_URL = "https://www.nexusmods.com/monsterhunterworld/mods/1982"


class App:
    def __init__(self, root, vrm_path=""):
        self.root = root
        root.title("VRoid Hunter %s" % __version__)
        root.minsize(620, 520)
        self.events = queue.Queue()
        self.busy = False
        self.files = None
        self.armor_ids = list(KNOWN_ARMOR)

        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="VRM model").grid(row=0, column=0, sticky="w", **pad)
        self.vrm = tk.StringVar(value=vrm_path)
        ttk.Entry(frm, textvariable=self.vrm).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.pick_vrm).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Game folder").grid(row=1, column=0, sticky="w", **pad)
        self.game = tk.StringVar(value=G.find_game() or "")
        ttk.Entry(frm, textvariable=self.game).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.pick_game).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Armor to replace").grid(row=2, column=0, sticky="w", **pad)
        self.armor = tk.StringVar(value=self._label(DEFAULT_ARMOR))
        self.armor_box = ttk.Combobox(frm, textvariable=self.armor, values=[self._label(a) for a in self.armor_ids])
        self.armor_box.grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Scan game", command=self.scan).grid(row=2, column=2, **pad)

        ttk.Label(frm, text="Height").grid(row=3, column=0, sticky="w", **pad)
        hf = ttk.Frame(frm)
        hf.grid(row=3, column=1, sticky="w", **pad)
        self.height = tk.IntVar(value=100)
        ttk.Spinbox(hf, from_=50, to=150, increment=1, textvariable=self.height, width=6).pack(side="left")
        ttk.Label(hf, text="% of the hunter's height (100 = exact fit)").pack(side="left", padx=6)

        self.hide_head = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Hide my hunter's own face and hair (recommended)",
                        variable=self.hide_head).grid(row=4, column=1, sticky="w", **pad)
        self.both = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Install for both female and male hunters",
                        variable=self.both).grid(row=5, column=1, sticky="w", **pad)

        bf = ttk.Frame(frm)
        bf.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        self.install_btn = ttk.Button(bf, text="Install", command=self.do_install)
        self.install_btn.pack(side="left", padx=4)
        ttk.Button(bf, text="Uninstall", command=self.do_uninstall).pack(side="left", padx=4)
        ttk.Button(bf, text="Export to folder...", command=self.do_export).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(bf, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=4)

        self.log_box = tk.Text(frm, height=14, wrap="word", state="disabled")
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(7, weight=1)

        self.log("1. Pick your .vrm file.  2. Pick the armor set to replace.  3. Press Install.")
        self.log("In game, equip that armor (or its layered version) on every slot to see your model.")
        self.check_loader()
        root.after(100, self.pump)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _label(armor_id):
        name = KNOWN_ARMOR.get(armor_id)
        return "%s  (%s)" % (armor_id, name) if name else armor_id

    def selected_armor(self):
        return self.armor.get().strip().split()[0].lower() if self.armor.get().strip() else DEFAULT_ARMOR

    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def check_loader(self):
        gd = self.game.get()
        if G.is_game_dir(gd) and not G.loader_installed(gd):
            self.log("WARNING: Stracker's Loader was not found in the game folder. "
                     "The game ignores nativePC mods without it: " + LOADER_URL)

    def pick_vrm(self):
        p = filedialog.askopenfilename(filetypes=[("VRM model", "*.vrm"), ("glTF binary", "*.glb"), ("All", "*.*")])
        if p:
            self.vrm.set(p)

    def pick_game(self):
        p = filedialog.askdirectory(title="Folder that contains MonsterHunterWorld.exe")
        if p:
            self.game.set(p)
            self.check_loader()

    def pump(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self.busy = False
                    self.progress.stop()
                    if payload:
                        messagebox.showinfo("VRoid Hunter", payload)
                elif kind == "error":
                    self.busy = False
                    self.progress.stop()
                    messagebox.showerror("VRoid Hunter", payload)
                elif kind == "armors":
                    self.armor_ids = payload
                    self.armor_box.configure(values=[self._label(a) for a in payload])
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def run(self, work):
        if self.busy:
            return
        self.busy = True
        self.progress.start(12)

        def target():
            try:
                msg = work(lambda m: self.events.put(("log", m)))
                self.events.put(("done", msg))
            except Exception as e:
                self.events.put(("log", traceback.format_exc()))
                self.events.put(("error", str(e)))

        threading.Thread(target=target, daemon=True).start()

    def _validate(self, need_vrm=True):
        gd = self.game.get()
        if not G.is_game_dir(gd):
            messagebox.showerror("VRoid Hunter", "Please choose the folder that contains MonsterHunterWorld.exe.")
            return False
        if need_vrm and not os.path.isfile(self.vrm.get()):
            messagebox.showerror("VRoid Hunter", "Please choose a .vrm file.")
            return False
        return True

    def _options(self):
        return Options(armor=self.selected_armor(),
                       genders=("f", "m") if self.both.get() else ("f",),
                       hide_face_and_hair=self.hide_head.get(),
                       height_scale=max(0.5, min(1.5, self.height.get() / 100.0)))

    # ---- actions ------------------------------------------------------------
    def scan(self):
        if not self._validate(need_vrm=False):
            return
        gd = self.game.get()

        def work(log):
            gf = GameFiles(gd, log)
            try:
                ids = armor_sets(gf)
            finally:
                gf.close()
            self.events.put(("armors", ids))
            log("Found %d armor sets." % len(ids))
            return None

        self.run(work)

    def _build(self, log):
        gf = GameFiles(self.game.get(), log)
        try:
            return build(self.vrm.get(), gf, self._options(), log)
        finally:
            gf.close()

    def do_install(self):
        if not self._validate():
            return
        gd = self.game.get()

        def work(log):
            result = self._build(log)
            G.install(gd, result.files, {"vrm": os.path.basename(self.vrm.get()),
                                         "armor": self.selected_armor()}, log)
            armor = self.selected_armor()
            extra = "" if G.loader_installed(gd) else \
                "\n\nStracker's Loader is not installed yet. Install it, or the game will not load the mod."
            return ("Installed! Equip the %s armor (or its layered version) on every slot to see "
                    "your model.%s" % (self._label(armor), extra))

        self.run(work)

    def do_uninstall(self):
        if not self._validate(need_vrm=False):
            return
        gd = self.game.get()
        self.run(lambda log: (G.uninstall(gd, log), "VRoid Hunter has been removed.")[1])

    def do_export(self):
        if not self._validate():
            return
        out = filedialog.askdirectory(title="Export the mod into this folder")
        if not out:
            return

        def work(log):
            result = self._build(log)
            G.export_folder(out, result.files)
            return "Exported %d files to %s" % (len(result.files), os.path.join(out, "nativePC"))

        self.run(work)


def main(vrm_path=""):
    root = tk.Tk()
    App(root, vrm_path)
    root.mainloop()
