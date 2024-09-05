
## Create dataset for FCNN
CBF_fun = CBF.get_function()
num_samples = 1000
train_data = np.zeros((num_samples,2))
train_labels = np.zeros((num_samples,1))

# Generate training points with labels
train_data = np.zeros((num_samples, 2))
train_labels = np.zeros((num_samples, 1))
for iter in range(num_samples):
    x = np.random.rand(1, 2) * 100
    y = CBF_fun(x)
    # print(f'x: {x}, y: {y}')

    train_data[iter, :] = x
    train_labels[iter] = y

# Generate testing points with labels
test_data = np.zeros((num_samples, 2))
test_labels = np.zeros((num_samples, 1))
for iter in range(num_samples):
    x = np.random.rand(1, 2) * num_samples
    y = CBF_fun(x)
    # print(f'x: {x}, y: {y}')

    test_data[iter, :] = x
    test_labels[iter] = y

train_data = np.double(train_data)
train_labels = np.double(train_labels)
test_data = np.double(test_data)
test_labels = np.double(test_labels)
    
# Parameters
params = {'batch_size': 50,
          'shuffle': True}

# Generators
training_set = Dataset(train_data, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_set, **params)

## Create FCNN (equivalent is to have L hidden layers of width N)
model = models.FCNet(nFeatures=CBF.dims[0], nHidden1=5, nOut=CBF.dims[1], mean=0, std=1, device=device, bn=False)

# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def train(dataloader, model, loss_fn, optimizer, losses, test_losses=[]):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X, 1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test after every batch
        test_losses = test(test_dataloader, model, loss_fn, test_losses)

        if batch % 25 == 0:  #25
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test(dataloader, model, loss_fn, losses):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, sgn=None)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses

epochs = 10
train_losses, test_losses = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses, test_losses)
    #test_losses = test(test_dataloader, model, loss_fn, test_losses)
print("Training Done!")

torch.save(model.state_dict(), "model_fc.pth")
print("Saved PyTorch Model State to model_xx.pth")

print(np.shape(train_losses))
print(np.shape(test_losses))

plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.yscale('log')
plt.show()