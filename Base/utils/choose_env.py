import numpy as np

env_list = [
    {
        'n_agents': 5,
        'x_dim': 25,
        'y_dim' :25,
        'n_cities': 2,
        'max_rails_in_city': 3,
        'min_malfunction_interval' : 50,
    },
    {
        'n_agents': 10,
        'x_dim': 30,
        'y_dim' :30,
        'n_cities': 2,
        'max_rails_in_city': 3,
        'min_malfunction_interval' : 100,
    },
    {
        'n_agents': 20,
        'x_dim': 30,
        'y_dim' :30,
        'n_cities': 3,
        'max_rails_in_city': 3,
        'min_malfunction_interval' : 200,
    },
    {
        'n_agents': 50,
        'x_dim': 20,
        'y_dim' :35,
        'n_cities': 3,
        'max_rails_in_city': 3,
        'min_malfunction_interval' : 500,
    },
    {
        'n_agents': 80,
        'x_dim': 35,
        'y_dim' :20,
        'n_cities': 5,
        'max_rails_in_city': 3,
        'min_malfunction_interval' : 800,
    },
    {
        'n_agents': 80,
        'x_dim': 35,
        'y_dim' :35,
        'n_cities': 5,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 800,
    },
    {
        'n_agents': 80,
        'x_dim': 40,
        'y_dim' :60,
        'n_cities': 9,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 800,
    },
    {
        'n_agents': 80,
        'x_dim': 60,
        'y_dim' :40,
        'n_cities': 13,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 800,
    },
    {
        'n_agents': 80,
        'x_dim': 60,
        'y_dim' :60,
        'n_cities': 17,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 800,
    },
    {
        'n_agents': 100,
        'x_dim': 80,
        'y_dim' :120,
        'n_cities': 21,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 1000,
    },
    {
        'n_agents': 100,
        'x_dim': 100,
        'y_dim' :80,
        'n_cities': 25,
        'max_rails_in_city': 4,
        'min_malfunction_interval' : 1000,
    }
]

def get_env(env_count):

    # return env_list[env%len(env_list)]
    return env_list[0]