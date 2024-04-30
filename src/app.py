import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb

from PIL import Image, ImageTk


class App(tk.Tk):
    # Phương thức khởi tạo
    def __init__(self, width_size=1200, height_size=680):
        super().__init__()

        # Đặt logo
        icon = tk.PhotoImage(file="../images/Logo.png")
        self.iconphoto(True, icon)
        # Khóa thay đổi kích thước cửa sổ
        self.resizable(False, False)
        # Đặt kích thước và vị trí hiển thị cửa sổ
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int(screen_width / 2 - width_size / 2)
        y = int(screen_height / 2 - height_size / 2)
        self.geometry(f"{width_size}x{height_size}+{x}+{y}")
        self.configure(background="#343746")
        # Tạo menu
        self.create_menu()

    def create_menu(self):
        # Tạo menu chính
        menu = tk.Menu(self)
        # Tạo menu File
        file_menu = tk.Menu(menu, tearoff=False)  # Không tách menu
        file_menu.add_command(
            label="Open...",
            accelerator="Ctrl+O",
            command=lambda: self.select_file(),
        )
        self.bind("<Control-o>", lambda event: self.select_file())
        file_menu.add_command(label="Save", accelerator="Ctrl+S", command=None)
        # self.bind("<Control-s>", lambda event: self.save_file())
        file_menu.add_separator()
        file_menu.add_command(
            label="Close Window",
            accelerator="Ctrl+W",
            command=lambda: self.quit(),
        )
        self.bind("<Control-w>", lambda event: self.quit())
        menu.add_cascade(label="File", menu=file_menu)
        # Tạo menu Run
        self.radio_model = tk.StringVar()
        self.radio_model.set("1")  # Mặc định chọn model 1
        self.radio_model.trace_add(
            "write", self.mark_radio_model
        )  # Theo dõi thay đổi giá trị radio_model
        run_menu = tk.Menu(menu, tearoff=False)
        run_menu.add_command(
            label="Start Prediction", accelerator="F5", command=None
        )
        # self.bind("<F5>", lambda event: None)
        run_menu.add_separator()
        run_menu.add_radiobutton(
            label="Model 1", variable=self.radio_model, value=1, command=None
        )
        run_menu.add_radiobutton(
            label="Model 2", variable=self.radio_model, value=2, command=None
        )
        menu.add_cascade(label="Run", menu=run_menu)
        # Tạo menu Help
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Video Demo", command=None)
        help_menu.add_command(label="About", command=None)
        menu.add_cascade(label="Help", menu=help_menu)
        # Hiển thị menu
        self.config(menu=menu)
        # Frame
        self.create_frame()

    def image_click(self, event):
        event.widget.config(bg="#1376F8")
        if self.current_selected_image_label is not None:
            self.current_selected_image_label.config(bg="#343746")
        image_path = event.widget.image_path
        self.image_name_label.config(text=image_path.split("/")[-1])
        img = Image.open(image_path)
        img.thumbnail((700, 700), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.selected_image_label.config(image=img)
        self.selected_image_label.image = img
        self.current_selected_image_label = event.widget

    def create_frame(self):
        self.current_selected_image_label = None  # Lưu label ảnh đang chọn
        # Tạo frame bên trái
        frame_left = tk.Frame(self, width=300)
        frame_left.grid(row=0, column=0, sticky="ns")
        # Tạo frame bên phải
        frame_right = tk.Frame(self)
        frame_right.grid(row=0, column=1, sticky="nsew")
        # Cấu hình frame
        self.rowconfigure(0, weight=1)  # Cấu hình hàng 0 chiếm hết không gian
        self.columnconfigure(
            1, weight=1
        )  # Cấu hình cột 1 chiếm hết không gian
        # Tạo frame cho hình ảnh đang chọn
        frame_selected_image = tk.Frame(frame_right, background="#343746")
        frame_selected_image.grid(row=0, column=0, sticky="nsew")
        self.image_name_label = tk.Label(
            frame_selected_image,
            fg="#FFFFFF",
            background="#343746",
            font=("Arial", 15),
        )
        self.image_name_label.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.selected_image_label = tk.Label(
            frame_selected_image, background="#343746"
        )
        self.selected_image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Tạo frame cho kết quả dự đoán
        frame_prediction = tk.Frame(
            frame_right,
            background="#282A36",
            highlightbackground="#FFFFFF",
            highlightthickness=1,
        )
        frame_prediction.grid(row=1, column=0, sticky="nsew")
        title_frame_prediction = tk.Label(
            frame_prediction,
            text="Result: ",
            fg="#FFFFFF",
            background="#282A36",
            font=("Arial", 15, "bold"),
        )
        title_frame_prediction.pack(side=tk.TOP, pady=10, padx=10, anchor="w")
        # Cấu hình frame
        frame_right.rowconfigure(
            0, weight=1
        )  # Cấu hình hàng 0 chiếm hết không gian
        frame_right.rowconfigure(
            1, weight=0
        )  # Cấu hình hàng 1 chiếm không gian nhỏ
        frame_right.columnconfigure(
            0, weight=1
        )  # Cấu hình cột 0 chiếm hết không gian
        # Tạo Canvas và Scrollbar cho frame hình ảnh bên trái
        self.canvas_images = tk.Canvas(frame_left, background="#343746")
        self.canvas_images.pack(side=tk.LEFT, fill=tk.BOTH)
        # Tạo frame con cho canvas
        self.frame_inside_canvas = tk.Frame(self.canvas_images, padx=1, pady=1)
        self.canvas_images.create_window(
            (0, 0), window=self.frame_inside_canvas, anchor="nw"
        )  # Tạo cửa sổ con trong canvas_images và đặt frame_inside_canvas vào cửa sổ con
        # Tạo scrollbar
        scrollbar_images = tk.Scrollbar(
            frame_left, orient="vertical", command=self.canvas_images.yview
        )
        scrollbar_images.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_images.config(yscrollcommand=scrollbar_images.set)

    def mark_radio_model(self, *args):
        print(self.radio_model.get())

    def select_file(self):
        file_types = [("Images", "*.png *.jpg *.jpeg *.bmp")]
        file_name = fd.askopenfilename(
            title="Select image",
            initialdir="../images/samples",
            filetypes=file_types,
        )
        if file_name:
            # Thêm ảnh vào frame_images_child
            img = Image.open(file_name)
            img.thumbnail(
                (280, 280), Image.LANCZOS
            )  # Thu nhỏ kích thước ảnh và Image.LANCZOS giảm răng cưa khi thu nhỏ
            img = ImageTk.PhotoImage(img)
            label = tk.Label(self.frame_inside_canvas, image=img, bg="#343746")
            label.image = img  # giữ tham chiếu
            label.image_path = (
                file_name  # Thêm thuộc tính image_path cho label
            )
            label.pack()
            label.bind(
                "<Button-1>", self.image_click
            )  # Bắt sự kiện click chuột trái vào ảnh
            # Cập nhật kích thước frame_inside_canvas sau khi thêm image
            self.frame_inside_canvas.update_idletasks()
            # Cập nhật vùng cuộn cho canvas
            self.canvas_images.config(
                scrollregion=self.canvas_images.bbox("all")
            )  # canvas_images.bbox("all") trả về tọa độ của hcn bao quanh nội dung của canvas

    def quit(self):
        message = "Do you want to close the window?"
        if mb.askyesno(
            message=message, icon=mb.QUESTION, title="Exit", parent=self
        ):
            self.destroy()


def main():
    app = App()
    app.title("Artificial Neural Network - ANN")
    app.mainloop()


if __name__ == "__main__":
    main()