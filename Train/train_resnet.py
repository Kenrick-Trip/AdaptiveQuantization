import torch

from torch import nn, optim

from Train.utils import evaluate_model, set_random_seeds, create_model, prepare_dataloader, save_model


def train(learning_rate,
          num_epochs,
          model_name,
          device=torch.device("cuda"),
          model_dir="saved_models",
          seed=0):
    model_filename = str(model_name) + "_mnist.pt",
    num_classes = 10  # MNIST

    set_random_seeds(random_seed=seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes, model_name=model_name)

    train_loader, test_loader = prepare_dataloader(num_workers=4, train_batch_size=128, eval_batch_size=256)

    # Train model.
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs)
    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)


def train_model(model,
                train_loader,
                test_loader,
                device,
                learning_rate,
                num_epochs):
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device,
                                                  criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
              .format(epoch,
                      train_loss,
                      train_accuracy,
                      eval_loss,
                      eval_accuracy))

    return model
