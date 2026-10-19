"""Tkinter front end: pick a .vrm, pick an armor set, press Install."""

import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from . import game as G
from . import oodle as O
from .build import DEFAULT_ARMOR, DEFAULT_SECOND_ARMOR, FULL_HELMET, KNOWN_ARMOR, Options, build, scan_armor
from .chunk import GameFiles

LOADER_URL = "https://www.nexusmods.com/monsterhunterworld/mods/1982"
NO_SECOND = "None"
INTRO = ("In Monster Hunter: World your hunter's look comes from the armor they wear. "
         "VRoid Hunter turns one armor set into your VRoid model: wear that set (or its layered "
         "armor) on all five slots and your hunter becomes your avatar, physics included. "
         "Nothing else in the game is changed, and it's only on your PC.")


class App:
    def __init__(self, root, vrm_path=""):
        self.root = root
        root.title("VRoid Hunter %s" % __version__)
        root.minsize(660, 560)
        self.events = queue.Queue()
        self.busy = False
        self.oodle_path = None
        self.armor_ids = list(KNOWN_ARMOR)
        self.coverage = {}

        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text=INTRO, wraplength=620, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(frm, text="VRM model").grid(row=1, column=0, sticky="w", **pad)
        self.vrm = tk.StringVar(value=vrm_path)
        ttk.Entry(frm, textvariable=self.vrm).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.pick_vrm).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Game folder").grid(row=2, column=0, sticky="w", **pad)
        self.game = tk.StringVar(value=G.find_game() or "")
        ttk.Entry(frm, textvariable=self.game).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.pick_game).grid(row=2, column=2, **pad)

        ttk.Label(frm, text="Armor to use").grid(row=3, column=0, sticky="w", **pad)
        self.armor = tk.StringVar(value=self._label(DEFAULT_ARMOR))
        self.armor_box = ttk.Combobox(frm, textvariable=self.armor, values=[self._label(a) for a in self.armor_ids])
        self.armor_box.grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Check sets", command=self.scan).grid(row=3, column=2, **pad)
        ttk.Label(frm, text="Pick a set you own as armor or layered armor. Sets marked 'full helmet' "
                            "hide your hunter's own face under the avatar's head.",
                  wraplength=520, foreground="#555").grid(row=4, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Label(frm, text="Second armor").grid(row=5, column=0, sticky="w", **pad)
        self.armor2 = tk.StringVar(value=self._label(DEFAULT_SECOND_ARMOR))
        self.armor2_box = ttk.Combobox(frm, textvariable=self.armor2,
                                       values=[NO_SECOND] + [self._label(a) for a in self.armor_ids])
        self.armor2_box.grid(row=5, column=1, sticky="ew", **pad)
        ttk.Label(frm, text="Optional: a second set that also becomes your avatar, e.g. for a different "
                            "layered look. Choose 'None' to only change the set above.",
                  wraplength=520, foreground="#555").grid(row=6, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Label(frm, text="Height").grid(row=7, column=0, sticky="w", **pad)
        hf = ttk.Frame(frm)
        hf.grid(row=7, column=1, sticky="w", **pad)
        self.height = tk.IntVar(value=100)
        ttk.Spinbox(hf, from_=50, to=150, increment=1, textvariable=self.height, width=6).pack(side="left")
        ttk.Label(hf, text="% of your hunter's height (100 = same height)").pack(side="left", padx=6)

        self.physics = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Physics for hair, skirts and accessories (from the VRM's spring bones)",
                        variable=self.physics).grid(row=8, column=1, sticky="w", **pad)
        self.keep = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Keep the VRoid model's own proportions (untick to stretch it to the hunter's build)",
                        variable=self.keep).grid(row=9, column=1, sticky="w", **pad)
        self.neck = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Thicken the neck slightly if your hunter's own neck would show through",
                        variable=self.neck).grid(row=10, column=1, sticky="w", **pad)
        self.both = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Make it work for both female and male hunters",
                        variable=self.both).grid(row=11, column=1, sticky="w", **pad)

        bf = ttk.Frame(frm)
        bf.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Button(bf, text="Install", command=self.do_install).pack(side="left", padx=4)
        ttk.Button(bf, text="Uninstall", command=self.do_uninstall).pack(side="left", padx=4)
        ttk.Button(bf, text="Export to folder...", command=self.do_export).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(bf, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=4)

        self.log_box = tk.Text(frm, height=12, wrap="word", state="disabled")
        self.log_box.grid(row=13, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(13, weight=1)

        self.check_loader()
        root.after(100, self.pump)
        if G.is_game_dir(self.game.get()):
            root.after(300, self.scan)

    # ---- helpers ----------------------------------------------------------
    def _label(self, armor_id):
        parts = [armor_id]
        if KNOWN_ARMOR.get(armor_id):
            parts.append(KNOWN_ARMOR[armor_id])
        cov = self.coverage.get(armor_id)
        if cov is not None and cov >= FULL_HELMET:
            parts.append("- full helmet")
        return "  ".join(parts)

    def selected_second_armor(self):
        text = self.armor2.get().strip()
        if not text or text == NO_SECOND:
            return ""
        return text.split()[0].lower()

    def selected_armor(self):
        text = self.armor.get().strip()
        return text.split()[0].lower() if text else DEFAULT_ARMOR

    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def check_loader(self):
        gd = self.game.get()
        if G.is_game_dir(gd) and not G.loader_installed(gd):
            self.log("Stracker's Loader isn't installed in your game folder yet. Monster Hunter: World "
                     "only loads mods with it, so install it once: " + LOADER_URL)

    def pick_vrm(self):
        p = filedialog.askopenfilename(filetypes=[("VRM model", "*.vrm"), ("glTF binary", "*.glb"), ("All", "*.*")])
        if p:
            self.vrm.set(p)

    def pick_game(self):
        p = filedialog.askdirectory(title="Folder that contains MonsterHunterWorld.exe")
        if p:
            self.game.set(p)
            self.check_loader()
            self.scan()

    def ensure_oodle(self):
        """Make sure the decompression DLL is available; ask the user if it isn't (main thread)."""
        if self.oodle_path:
            return True
        found = O.find_oodle(self.game.get(), log=self.log)
        if found:
            self.oodle_path = found
            return True
        answer = messagebox.askyesnocancel(
            "One extra file needed",
            "Monster Hunter: World keeps its files compressed with a library called Oodle "
            "(oo2core_8_win64.dll). The game only includes an older copy that can't open its "
            "current files, so VRoid Hunter needs the newer one to read your hunter's skeleton.\n\n"
            "It wasn't found in your other installed games.\n\n"
            "Yes: download it now. (It's publicly hosted on the update server of Warframe, another "
            "game that uses it; the community MHW tools get it from the same place. The file is "
            "checked before use.)\n"
            "No: choose the file yourself, if you have it from another game.\n"
            "Cancel: stop.")
        if answer is None:
            return False
        if answer:
            try:
                self.oodle_path = O.download_oodle(self.log)
                return True
            except Exception as e:
                messagebox.showerror("VRoid Hunter", "The download failed: %s\n\nYou can choose the file "
                                                     "yourself instead." % e)
        path = filedialog.askopenfilename(title="Choose oo2core_8_win64.dll (or oo2core_9_win64.dll)",
                                          filetypes=[("Oodle library", "oo2core_*_win64.dll"), ("DLL", "*.dll")])
        if not path:
            return False
        self.oodle_path = O.remember(path)
        return True

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
                    self.apply_scan(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def apply_scan(self, results):
        self.coverage = {a: c for a, c in results}
        full = [a for a, c in results if c is not None and c >= FULL_HELMET]
        rest = [a for a, c in results if a not in full]
        self.armor_ids = full + rest
        self.armor_box.configure(values=[self._label(a) for a in self.armor_ids])
        self.armor2_box.configure(values=[NO_SECOND] + [self._label(a) for a in self.armor_ids])
        second = self.selected_second_armor()
        if second:
            self.armor2.set(self._label(second))
        current = self.selected_armor()
        if self.coverage.get(current, 0) is None or self.coverage.get(current, 0) < FULL_HELMET:
            preferred = [a for a in full if a in KNOWN_ARMOR] or full
            if preferred:
                current = preferred[0]
        self.armor.set(self._label(current))
        self.log("Found %d armor sets, %d with a full helmet. Selected %s." % (
            len(results), len(full), self._label(current)))

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

    def _validate(self, need_vrm=True, need_files=True):
        gd = self.game.get()
        if not G.is_game_dir(gd):
            messagebox.showerror("VRoid Hunter", "Please choose the folder that contains MonsterHunterWorld.exe.")
            return False
        if need_vrm and not os.path.isfile(self.vrm.get()):
            messagebox.showerror("VRoid Hunter", "Please choose your .vrm file.")
            return False
        if need_files and not self.ensure_oodle():
            return False
        return True

    def _options(self):
        return Options(armor=self.selected_armor(), second_armor=self.selected_second_armor(),
                       genders=("f", "m") if self.both.get() else ("f",),
                       physics=self.physics.get(),
                       keep_proportions=self.keep.get(),
                       cover_neck=self.neck.get(),
                       height_scale=max(0.5, min(1.5, self.height.get() / 100.0)))

    def _files(self, log):
        return GameFiles(self.game.get(), log, oodle_path=self.oodle_path)

    # ---- actions ------------------------------------------------------------
    def scan(self):
        if not self._validate(need_vrm=False):
            return

        def work(log):
            gf = self._files(log)
            try:
                results = scan_armor(gf, "f", log)
            finally:
                gf.close()
            self.events.put(("armors", results))
            return None

        self.run(work)

    def _build(self, log):
        gf = self._files(log)
        try:
            return build(self.vrm.get(), gf, self._options(), log)
        finally:
            gf.close()

    def do_install(self):
        if not self._validate():
            return
        gd = self.game.get()
        armor = self.selected_armor()
        second = self.selected_second_armor()
        sets = self._label(armor) + ("" if not second or second == armor else " or " + self._label(second))

        def work(log):
            result = self._build(log)
            G.install(gd, result.files, {"vrm": os.path.basename(self.vrm.get()), "armor": armor,
                                         "second_armor": second}, log)
            extra = "" if G.loader_installed(gd) else \
                "\n\nStracker's Loader isn't installed yet. Install it once, or the game won't load the mod."
            return ("Installed! In game, wear the %s set (or its layered armor) on all five slots, with "
                    "'Show headgear' on, to see your avatar.%s" % (sets, extra))

        self.run(work)

    def do_uninstall(self):
        if not self._validate(need_vrm=False, need_files=False):
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
