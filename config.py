
config = {
    "on_policy_mc": {
        'CartPole-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 100,
            "save_name": "monte_carlo_cartpole.pt",
            "update_type": "avg",
        }
    },
    "q_learning": {
        'CartPole-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 100,
            "save_name": "q_learning_cartpole.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'MountainCar-v0': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 10,
            "save_name": "q_learning_mountaincar.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'Acrobot-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 10,
            "save_name": "q_learning_acrobot.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'Taxi-v3': {
            "epsilon": 0.1,
            "gamma": 1,
            "init_val": 10,
            "save_name": "q_learning_acrobot.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        }
    },
    'sarsa': {
        'CartPole-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 100,
            "save_name": "sarsa_cartpole.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'MountainCar-v0': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 10,
            "save_name": "sarsa_mountaincar.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'Acrobot-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 10,
            "save_name": "sarsa_acrobot.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        },
        'Taxi-v3': {
            "epsilon": 0.1,
            "gamma": 1,
            "init_val": 10,
            "save_name": "sarsa_acrobot.pt",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.0001
        }
    },
    "DynaQ": {
        'Acrobot-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 0,
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 1e-5,
            "buffer_size": 10 ** 10,
            "n_planning_steps": 10,
            "save_name": "dynaq_acrobot.pt",
            "buffer_save_name": "dynaq_acrobot_buffer.pt"
        },
        'CartPole-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 10,
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 1e-5,
            "buffer_size": 10 ** 10,
            "n_planning_steps": 50,
            "save_name": "dynaq_cartpole.pt",
            "buffer_save_name": "dynaq_cartpole_buffer.pt"
        },
        "Taxi-v3": {
            "epsilon": 0.1,
            "gamma": 1,
            "init_val": 0,
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 1e-5,
            "buffer_size": 10 ** 10,
            "n_planning_steps": 100,
            "save_name": "dynaq_taxi.pt",
            "buffer_save_name": "dynaq_taxi_buffer.pt"
        }
    }

}