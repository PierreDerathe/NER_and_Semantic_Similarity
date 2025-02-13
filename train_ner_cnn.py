import torch
from preprocess_conll import preprocess_conll
from cnn_classification import CNNForNER, load_and_preprocess_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='QUAERO_FrenchMed/EMEA/EMEAtrain_layer1_ID.conll', 
                        help="Training data in CONLL format")
    parser.add_argument("--valid", default='QUAERO_FrenchMed/EMEA/EMEAdev_layer1_ID.conll',
                        help="Validation data in CONLL format")
    parser.add_argument("--test", default='QUAERO_FrenchMed/EMEA/EMEAtest_layer1_ID.conll',
                        help="Test data in CONLL format")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
    args = parser.parse_args()

    # Preprocess CONLL files
    train_df, label_encoder = preprocess_conll(args.train)
    valid_df, _ = preprocess_conll(args.valid)
    test_df, _ = preprocess_conll(args.test)

    # Get number of unique NER tags
    num_classes = len(label_encoder.classes_)

    # Load and preprocess data using existing function
    train_x, train_y, valid_x, valid_y, test_x, test_y, vocab_index = load_and_preprocess_data(
        train_df, valid_df, test_df, sequence_length=128, max_vocab=32000
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNForNER(
        vocab_size=len(vocab_index),
        embedding_size=100,
        class_size=num_classes
    ).to(device)

    # Training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, num_classes), target.view(-1))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.numel()
        
        print(f'Epoch {epoch+1}: Validation Accuracy = {100. * correct / total:.2f}%')

if __name__ == "__main__":
    main()