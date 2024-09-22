import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import colorchooser
import color_classification as predict

# Tạo cửa sổ chính
root = tk.Tk()
root.geometry("600x400+0+0")
root.title("COLOR_REGCONITION")
root.configure(background="white")
root.resizable(False, False)


def browse_file():
    file_path = filedialog.askopenfilename()
    path_var.set(file_path)
    entry_path.delete(0, tk.END)  # Xóa nội dung hiện tại của Entry
    entry_path.insert(0, file_path)  # Gán đường dẫn vào Entry


def classify():
    print(path_var.get())
    predict.sol(path_var.get())

def clear():
    path_var.set('')
    entry_path.delete(0, tk.END)
    return 


# Tạo một biến StringVar để lưu trữ đường dẫn được chọn
path_var = tk.StringVar()

# Tạo một nhãn để hiển thị đường dẫn
label = tk.Label(root,
                 text="This is an app for Color_recognition. \n You must upload an image to detect.\n Try it.",
                 background="white")
label.grid(row=0, column=1)

# Tạo một Entry để hiển thị đường dẫn đã chọn
label_path = tk.Label(root,
                      text="File :",
                      background="white").grid(row=1, column=0)
entry_path = tk.Entry(root, textvariable=path_var, width=80)
entry_path.grid(row=1, column=1)

# # Tạo một cửa sổ để
# label_detect = tk.Label(root,
#                         text="Detect :",
#                         background="beige").grid(row=2, column=0)
# text_widget = tk.Text(root, width=50, height=10, padx=1, pady=1, state="disabled", background="gray")
# text_widget.grid(row=2, column=1, padx=70)

# Tạo 3 nút upload-detect-clear
button_upload = tk.Button(root, text="Upload", background="white",
                          width=10, height=2, padx=50, pady=5,
                          command=browse_file)
button_upload.grid(row=3, column=0, padx=10)

button_detect = tk.Button(root, text="Classify", background="white", width=10,
                          height=2, padx=5, pady=5,
                          command=classify)
button_detect.grid(row=3, column=1)


button_clear = tk.Button(root, text="Clear", background="white",
                         width=10, height=2, padx=50, pady=5,
                         command=clear)
button_clear.grid(row=3, column=2, padx=5)

# Thiết lập lưới cho giao diện
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.mainloop()
