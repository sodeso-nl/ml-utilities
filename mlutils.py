import matplotlib as plt

def plot_history(history, figsize=(10, 6)):
  plt.figure(figsize=figsize)

  # Plot the traiing loss and accuracy
  plt.plot(history.history['loss'], label='Training loss', color='#0000FF', linewidth=1.5)
  if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation loss', color='#00FF00', linewidth=1.5)

  # Plot the learning rate
  if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning rate', color='#000000', linewidth=1.5, linestyle='--')

  plt.title('Loss', size=20)
  plt.xticks(history.epoch)
  plt.xlabel('Epoch', size=14)
  plt.legend()

  # Start a new figure
  plt.figure(figsize=figsize, facecolor='#FFFFFF')

  # Plot the validation loss and accuracy
  plt.plot(history.history['accuracy'], label='Training accuracy', color='#0000FF', linewidth=1.5)
  if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation accuracy', color='#00FF00', linewidth=1.5)

  # Plot the learning rate
  if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning rate', color='#000000', linewidth=1.5, linestyle='--')
  plt.title('Accuracy', size=20)
  plt.xticks(history.epoch)
  plt.xlabel('Epoch', size=14)
  plt.legend()

  plt.show()