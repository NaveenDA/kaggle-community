import torch
import pandas as pd
import numpy as np
from cnn import DigitRecognizer, MNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def predict_and_submit():
    # Load the best model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('../datasets/digit-recognizer/test.csv')
    X_test = test_data.values

    # Create test dataset and dataloader
    test_dataset = MNISTDataset(X_test, np.zeros(len(X_test)), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define test-time augmentations
    tta_transforms = [
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.ToTensor(),
    ]

    # Generate predictions with test-time augmentation
    print("Generating predictions...")
    predictions = []
    num_tta = 10  # Increased number of TTA iterations
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_predictions = []
            
            # Original prediction
            outputs = model(inputs)
            batch_predictions.append(outputs)
            
            # TTA predictions
            for _ in range(num_tta):
                # Apply random augmentation
                aug_inputs = inputs.cpu()
                for i in range(len(aug_inputs)):
                    img = aug_inputs[i].view(28, 28)
                    for transform in tta_transforms:
                        img = transform(img)
                    aug_inputs[i] = img.view(1, 28, 28)
                aug_inputs = aug_inputs.to(device)
                
                # Get prediction
                outputs = model(aug_inputs)
                batch_predictions.append(outputs)
            
            # Average predictions
            avg_outputs = torch.stack(batch_predictions).mean(0)
            _, predicted = torch.max(avg_outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    submission.to_csv('../datasets/digit-recognizer/sample_submission.csv', index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    predict_and_submit() 