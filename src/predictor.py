import cv2

class Visualization:
    def __init__(self):
        pass
    def cfg(self, model):
        print("Configuring model...")
        print(f"Model: {model}")
    def run(self, image_path):
        print("Running...")
        print(f"Image path: {image_path}")

        img = cv2.imread(image_path[0])
        cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 0), 2)
        cv2.putText(img, "DEMO", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img