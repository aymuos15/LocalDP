from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)

            # # plot the inputs, outputs and targets side by side
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
            # ax[0].set_title("Input")
            # ax[1].imshow(outputs[0, 0].detach().cpu().numpy())
            # ax[1].set_title("Output")
            # ax[2].imshow(targets[0, 0].cpu().numpy())
            # ax[2].set_title("Target")
            # plt.show()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Loss: {total_loss/len(dataloader):.4f}")