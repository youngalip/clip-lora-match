from transformers import CLIPModel, CLIPProcessor

def main():
    model_name = "openai/clip-vit-base-patch32"
    print("Loading model...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Done loading!")

if __name__ == "__main__":
    main()
