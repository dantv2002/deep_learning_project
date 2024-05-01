import cv2

class Visualization:
    def __init__(self):
        pass
    def cfg(self, model):
        print("Configuring model...")
        print(f"Model: {model}")
        self.model = model
    
    def run(self, images)->list:
        print("Running...")
        batch_image = [cv2.imread(image["image_path"]) for image in images]
        image_result=self.predict(batch_image)
        results = []
        for i in range(len(images)):
            
            results.append({
                "id": images[i]["id"],
                "image": image_result[i]
            })
        return results
    def predict(self, batch_image):
        print("Predicting...")
        results = []
        for image in batch_image:
            results.append(self.draw(image))
        return results
    def draw(self, image):
        print("Drawing...")
        if(self.model == 1):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(image, (0, 0), (100, 100), color, 2)
        cv2.putText(image, "DEMO", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image