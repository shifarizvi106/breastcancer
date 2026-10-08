import time
from pathlib import Path
from PIL import Image, ImageOps

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from app.inference_cancer import InferenceModel

# ----------------------------
# CONFIG
# ----------------------------
BG        = "#f4dad4"
ACCENT    = "#dd8c96"
ACCENT_HOVER = "#c97a85"
TEXT_DARK = "#2e4060"
WHITE     = "#ffffff"

# ----------------------------
# Helper functions
# ----------------------------
def format_percent(x: float) -> str:
    return f"{x*100:.1f}%"

def resize_preview(pil_img: Image.Image, max_side=400) -> Image.Image:
    img = pil_img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ----------------------------
# App
# ----------------------------
class CuteMedApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Breast Cancer Predictior")
        self.geometry("950x640")
        self.minsize(900, 600)

        ctk.set_appearance_mode("light")
        self.configure(fg_color=BG)

        self.model = None
        self.current_img = None

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.build_left_panel()
        self.build_right_panel()

    def build_left_panel(self):
        panel = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=16)
        panel.grid(row=0, column=0, sticky="nsw", padx=14, pady=14)
        panel.grid_propagate(False)
        panel.configure(width=300)

        ctk.CTkLabel(panel, text="Breast Cancer Detection",
                     font=("Times New Roman", 22, "bold"),
                     text_color=TEXT_DARK).pack(pady=(30, 20))

        self.load_btn = ctk.CTkButton(panel, text="Load Model", command=self.load_model,
                                      fg_color=ACCENT, hover_color=ACCENT_HOVER, height=50,
                                      font=("Times New Roman", 16, "bold"), text_color=WHITE)
        self.load_btn.pack(pady=(0, 10), padx=20, fill="x")

        self.model_status = ctk.CTkLabel(panel, text="Model not loaded",
                                          text_color="#666", font=("Times New Roman", 12))
        self.model_status.pack(pady=(0, 30))

        self.upload_btn = ctk.CTkButton(panel, text="Upload Image", command=self.open_image,
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER, height=50,
                                        font=("Times New Roman", 16, "bold"), text_color=WHITE)
        self.upload_btn.pack(pady=(0, 10), padx=20, fill="x")

        self.pred_btn = ctk.CTkButton(panel, text="Predict", command=self.run_predict,
                                      fg_color=ACCENT, hover_color=ACCENT_HOVER, height=50,
                                      font=("Times New Roman", 16, "bold"), text_color=WHITE)
        self.pred_btn.pack(pady=(0, 30), padx=20, fill="x")

        res_box = ctk.CTkFrame(panel, fg_color=BG, corner_radius=12)
        res_box.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(res_box, text="Result", font=("Times New Roman", 18, "bold"),
                     text_color=TEXT_DARK).pack(pady=(15, 10))

        self.label_out = ctk.CTkLabel(res_box, text="—", font=("Times New Roman", 32, "bold"),
                                       text_color=TEXT_DARK)
        self.label_out.pack(pady=(0, 15))

    def build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 14), pady=14)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(right, text="Open an image to preview",
                                          text_color="#777", font=("Times New Roman", 14))
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    def load_model(self):
        try:
            mp = "models/best_cancer_model.pth"
            t0 = time.time()
            self.model = InferenceModel(mp)
            dt = time.time() - t0
            self.model_status.configure(text=f"Model loaded ({dt:.2f}s)", text_color="#12a454")
            self.load_btn.configure(text="Model Loaded", state="disabled")
        except Exception as e:
            self.model_status.configure(text=f"Error: {str(e)}", text_color="#ff4444")
            messagebox.showerror("Error loading model", str(e))

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select histopathology image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            self.current_img = pil
            self.show_preview(pil)
            self.clear_outputs()
        except Exception as e:
            messagebox.showerror("Image error", f"Could not open image:\n{e}")

    def show_preview(self, pil_img: Image.Image):
        img = resize_preview(pil_img, max_side=520)
        self._tkimg = ctk.CTkImage(light_image=img, size=img.size)
        self.preview_label.configure(image=self._tkimg, text="")

    def clear_outputs(self):
        self.label_out.configure(text="—", text_color=TEXT_DARK)

    def run_predict(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Load a model first.")
            return
        if self.current_img is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return

        self.pred_btn.configure(text="Predicting...", state="disabled")
        self.label_out.configure(text="Analyzing...", text_color="#666")
        self.update()

        try:
            out = self.model.predict(self.current_img, threshold=0.5)
            label = out["label"]
            conf = out["confidence"]

            is_malignant = "maglinant" in label.lower()
            display = "Malignant" if is_malignant else "Benign"
            color = ACCENT if is_malignant else "#12a454"
            self.label_out.configure(text=f"{display}\n{format_percent(conf)}", text_color=color)

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
        finally:
            self.pred_btn.configure(text="Predict", state="normal")


if __name__ == "__main__":
    app = CuteMedApp()
    app.mainloop()
