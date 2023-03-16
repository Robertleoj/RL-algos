
config = {
    "on_policy_mc": {
        'CartPole-v1': {
            "epsilon": 0.05,
            "gamma": 0.99,
            "init_val": 100,
            "save_name": "on_policy_mc_cartpole.pt",
            # "update_type": "avg",
            "update_type": "lr",
            "lr": 1,
            "lr_decay": 0.01
        }
    },
    "off_policy_mc": {
        'CartPole-v1': {
            'epsilon': 0.05,
            "gamma": 0.999,
            "init_val": 10,
            "save_name": "off_policy_mc_cartpole.pt",
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
    "n_step_sarsa":{
        'CartPole-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 200,
            "save_name": "n_step_sarsa_cartpole.pt",
            # "update_type": "lr",
            # "lr": 1,
            # "lr_decay": 0.0001,
            "update_type":"exp_avg",
            "alpha": 0.002,
            "n": 25
        } ,
        'FlappyBird-v0': {
            "epsilon": 0.005,
            "gamma": 0.999,
            "init_val": 5,
            "save_name": "n_step_sarsa_flappybird.pt",
            # "update_type": "lr",
            # "lr": 1,
            # "lr_decay": 0.0001,
            # "update_type":"exp_avg",
            # "alpha": 0.2,
            "update_type": "avg",
            "n": 50
        },
        "Acrobot-v1": {
            "epsilon": 0.001,
            "gamma": 0.999,
            "init_val": 0,
            "save_name": "n_step_sarsa_acrobot.pt",
            "update_type": 'avg',
            # "update_type": "lr",
            # "lr": 1,
            # "lr_decay": 0.0001,
            # "update_type":"exp_avg",
            # "alpha": 0.1,
            "n": 25
        }
    },

    "DynaQ": {
        'Acrobot-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 0,
            "update_type": "avg",
            # "lr": 1,
            # "lr_decay": 1e-5,
            "save_name": "dynaq_acrobot.pt",
            "buffer_save_name": "dynaq_acrobot_buffer.pt",
            "n_planning_steps": 10,
            "buffer_size": 10** 10
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
            "buffer_save_name": "dynaq_cartpole_buffer.pt",
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
            "buffer_save_name": "dynaq_taxi_buffer.pt",
        },
        "FlappyBird-v0": {
            "epsilon": 0.01,
            "gamma": 0.999,
            "init_val": 5,
            "update_type": "avg",
            # "update_type": "exp_avg",
            # "alpha": 0.1,
            # "update_type": "lr",
            # "lr": 1,
            # "lr_decay": 1e-5,
            "buffer_size": 10 ** 10,
            "n_planning_steps": 100,
            "save_name": "dynaq_flappy.pt",
            "buffer_save_name": "dynaq_flappy_buffer.pt",
        }
    },
    "DynaQ_prioritized": {
        'Acrobot-v1': {
            "epsilon": 0.1,
            "gamma": 0.999,
            "init_val": 0,
            "update_type": "avg",
            "save_name": "dynaq_prioritized_acrobot.pt",
            "buffer_save_name": "dynaq_prioritized_acrobot_buffer.pt",
            "pq_threshold": 0.2,
            "n_planning_steps": 5,
            "reverse_samples": 5
        },
        'FlappyBird-v0': {
            "epsilon": 0.001,
            "gamma": 0.999,
            "init_val": 5,
            "update_type": "avg",
            "save_name": "dynaq_prioritized_acrobot.pt",
            "buffer_save_name": "dynaq_prioritized_acrobot_buffer.pt",
            "pq_threshold": 0.2,
            "n_planning_steps": 10,
            "reverse_samples": 10
        }
    }
}