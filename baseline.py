from PPO_inv_paratune import InvEnv
import numpy as np
from tqdm import tqdm

import sys

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    
    def flush(self):
        pass

sys.stdout = Logger("log.txt")


poisson_lambda = 5.
holding_cost = 1.
query_max = 20
inv_max = 100
test_max_episode_length = 20

for shortage_penalty in [4, 9, 19]:
    for lead in [1, 2, 3, 4, 6, 8, 10]:
        print(f"shortage_penalty: {shortage_penalty}")
        print(f"lead: {lead}")
        env = InvEnv(poisson_lambda, lead, query_max=query_max,
            query_cost=0., holding_cost=holding_cost, shortage_penalty=shortage_penalty, 
            inv_max=inv_max, max_episode_length=test_max_episode_length
        )
        # constant
        """
        avg_r = 0.
        for _ in (pbar:=tqdm(range(10000), ncols=0)):
            env.reset()
            total_r = 0.
            for i in range(test_max_episode_length):
                a = poisson_lambda
                r = env.step(a)[1]
                total_r += r
            avg_r += total_r
            pbar.set_description(f"avg r: {avg_r/(_+1)/test_max_episode_length}")
        """
        # capped base stock
        capped_base_stock_S_list = np.arange(1,60)
        capped_base_stock_r_list = np.arange(1,20)
        best_avg_r = -1e9
        best_S = 0
        best_r = 0
        for capped_base_stock_S in capped_base_stock_S_list:
            for capped_base_stock_r in capped_base_stock_r_list:
                avg_r = 0.
                for _ in (pbar:=tqdm(range(1000), ncols=0)):
                    env.reset()
                    total_r = 0.
                    for i in range(test_max_episode_length):
                        a = int(min(max(capped_base_stock_S-np.sum(env.inv_pipeline), 0), capped_base_stock_r))
                        r = env.step(a)[1]
                        total_r += r
                    avg_r += total_r
                    pbar.set_description(f"avg r: {avg_r/(_+1)/test_max_episode_length}")
                if avg_r/(_+1)/test_max_episode_length > best_avg_r:
                    best_avg_r = avg_r/(_+1)/test_max_episode_length
                    best_S = capped_base_stock_S
                    best_r = capped_base_stock_r     
        print("best_S: ", best_S)
        print("best_r: ", best_r)
        print("best_avg_r: ", best_avg_r)
        
