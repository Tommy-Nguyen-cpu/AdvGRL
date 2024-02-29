import clip
import torch
from PIL import Image

class CLIP_Classifier:
    def __init__(self):

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)

    def predict(self, image_path, classes, top):
        # Prepare the inputs
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(self.device)

        if top > len(classes):
            print("The number of top predictions you want is higher than the number of classes provided.")
            return []
        
        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top)

        # Print the result
        # print("\nTop predictions:\n")
        predictions = []
        for value, index in zip(values, indices):
            predictions.append(("", classes[index], value.item()))
            # print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
            # print("value: " + str(value.item()))
        return predictions