import pickle
import matplotlib.pyplot as plt


with open('mean_loss.pkl', 'rb') as f:
    origin = pickle.load(f)
    f.close()


with open('mean_loss_update.pkl', 'rb') as f:
    update = pickle.load(f)
    f.close()


plt.plot(range(origin.__len__()), origin,label='Original_Model', color='red')
plt.plot(range(update.__len__()), update,label='Update_Model', color='orange')
plt.legend()
plt.show()