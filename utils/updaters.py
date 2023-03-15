class LRUpdater:
    def __init__(self, lr, lr_decay):
        self.lr = lr
        self.lr_decay = lr_decay

    def __call__(self, curr_val, new_val, num_encounters):
        lr = self.lr * (1 / (1 + self.lr_decay * num_encounters))

        return curr_val + lr * (new_val - curr_val)

class AvgUpdater:

    def __call__(self, curr_val, new_val, num_encounters):
        return (num_encounters * curr_val + new_val) / (num_encounters + 1)

class ExpAvgUpdater:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, curr_val, new_val, num_encounters):
        return self.alpha * new_val + (1 - self.alpha) * curr_val
    

def get_updater(conf):
    update_type = conf['update_type']

    if update_type == "avg":
        return AvgUpdater()
    elif update_type == 'lr':
        return LRUpdater(conf['lr'], conf['lr_decay'])
    elif update_type =="exp_avg":
        return ExpAvgUpdater(conf['alpha'])
    else:
        raise Exception("Invalid update type")

