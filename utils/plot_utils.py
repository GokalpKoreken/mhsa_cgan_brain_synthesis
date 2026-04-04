import matplotlib.pyplot as plt

def plot_losses_generator(mean_train_losses, mean_val_losses, filename):
    # Line plot for training progress in terms of losses
    plt.figure(figsize=(12,6))
    plt.plot(mean_train_losses, '-'),
    plt.plot(mean_val_losses, '-'),
    plt.xlabel('Epoch'),
    plt.ylabel('Loss'),
    plt.legend(['Train', 'Validation']),
    plt.title('Training vs. Validation Mean Losses'),
    plt.savefig(filename)
    plt.close() 
    
def plot_losses_cgan(mean_train_losses_G, mean_train_losses_D, filename):
    # Line plot for training progress in terms of losses
    plt.figure(figsize=(12,6))
    plt.plot(mean_train_losses_G, '-', color='blue'),
    #plt.plot(mean_val_losses_G, '--', color='lightskyblue'),
    plt.plot(mean_train_losses_D, '-', color='red'),
    #plt.plot(mean_val_losses_D, '--', color='lightcoral'),
    plt.xlabel('Epoch'),
    plt.ylabel('Loss'),
    plt.legend(['Generator (train)', 'Discriminator (train)']),
    plt.title('Training Mean Losses (Generator and Discriminator)'),
    plt.savefig(filename)
    plt.close()
    
def plot_Dlosses_cgan(mean_train_losses_D, filename):
    # Line plot for training progress in terms of losses
    plt.figure(figsize=(12,6))
    plt.plot(mean_train_losses_D, '-', color='red'),
    #plt.plot(mean_val_losses_D, '--', color='lightcoral'),
    plt.xlabel('Epoch'),
    plt.ylabel('Loss'),
    plt.legend(['Discriminator (train)']),
    plt.title('Training Mean Losses (Discriminator)'),
    plt.savefig(filename)
    plt.close() 
    
def plot_losses_reconst(mean_train_l1_loss, filename):
    # Line plot for training progress in terms of l1 loss
    plt.figure(figsize=(12,6))
    plt.plot(mean_train_l1_loss, '-', color='blue'),
    #plt.plot(mean_val_l1_loss, '-', color='red'),
    plt.xlabel('Epoch'),
    plt.ylabel('Loss'),
    plt.legend(['Train']),
    plt.title('Training Mean L1 loss (Generator - normalized loss)'),
    plt.savefig(filename)
    plt.close() 
    
def plot_all_losses_cgan(mean_train_gan_loss, mean_train_l1_loss, mean_train_pl_loss, mean_train_dreal, mean_train_dfake, filename, train_or):
    # Line plot for training progress in terms of l1 loss
    plt.figure(figsize=(12,6))
    plt.plot(mean_train_gan_loss, '-', color='blue'),
    plt.plot(mean_train_l1_loss, '-', color='orange'),
    plt.plot(mean_train_pl_loss, '-', color='pink'),
    plt.plot(mean_train_dreal, '-', color='green'),
    plt.plot(mean_train_dfake, '-', color='red'),
    plt.xlabel('Epoch'),
    plt.ylabel('Loss'),
    plt.legend(['G_GAN', 'G_L1', 'G_PL', 'D_real', 'D_fake']),
    plt.title(f'cGAN loss over time - {train_or}'),
    plt.savefig(filename)
    plt.close() 
    
    
def plot_metrics(fold_metrics, filename_metric, ylabel_metric, title_metric):
    # Bar plot for validation metrics per fold
    plt.figure(figsize=(6,6))
    folds = range(1, len(fold_metrics)+1)
    plt.bar(folds, fold_metrics, color='skyblue', edgecolor='black')
    plt.xlabel("Fold")
    plt.ylabel(ylabel_metric)
    plt.title(title_metric)
    plt.savefig(filename_metric)
    plt.close() 
    
def plot_mean_metrics(mean_metric, filename_metric, ylabel_metric, title_metric):
    # Line plot for metrics values progress through images for each fold
    #plt.style.use('ggplot')
    plt.figure(figsize=(12,6))
    plt.plot(mean_metric, '-', color='orange'),
    plt.xlabel("Epoch")
    plt.ylabel(ylabel_metric)
    plt.title(title_metric)
    plt.savefig(filename_metric)
    plt.close() 
        
    