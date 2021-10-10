from tqdm import tqdm
import torch

def train(train_loader, model, optimizer):
    train_loss = 0

    for i, inputs in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        model.train()
        inputs = {key: val.reshape(val.shape[0], -1) for key, val in inputs.items()}
        outputs = model(**inputs)
        loss = outputs[1]
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss


def evaluate(valid_loader, model):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(valid_loader)):
            inputs = {key: val.reshape(val.shape[0], -1) for key, val in inputs.items()}
            outputs = model(**inputs)
            loss = outputs[1]

            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    return valid_loss
