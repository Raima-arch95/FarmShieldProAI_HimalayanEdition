import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.
    Returns the average training loss.
    """

    model.train()

    running_loss = 0.0

    for batch_index, batch in enumerate(dataloader):

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Batch {batch_index + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(dataloader)

def evaluate_model(
    model,
    dataloader,
    device,
):
    """
    Run inference on the evaluation dataset.

    Returns:
        predictions (list[int])
        labels (list[int])
    """

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            targets = batch["label"].to(device)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())

            labels.extend(targets.cpu().tolist())

    return predictions, labels

def validate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Evaluate the model on the validation dataset.
    """

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    average_loss = running_loss / len(dataloader)

    accuracy = 100 * correct / total

    return average_loss, accuracy
