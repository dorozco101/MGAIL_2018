import numpy as np
import os
from environment import Environment
from driver import Driver


def dispatcher(env):

    driver = Driver(env)
    driver_itrs = []

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats

            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)
            
            #save running rewards and stds
            driver_itrs.append(driver.itr)
            driver.reward_mean_Arr.append(driver.reward_mean)
            driver.reward_std_Arr.append(driver.reward_std)
            
            np.savetxt("reward_itrs_ip.csv", np.array(driver_itrs), delimiter=",")
            np.savetxt("reward_means_ip.csv", np.array(driver.reward_mean_Arr), delimiter=",")
            np.savetxt("reward_stds_ip.csv", np.array(driver.reward_std_Arr), delimiter=",")
            
            np.save('reward_itrs_ip.npy',np.array(driver_itrs))
            np.save('reward_means_ip.npy',np.array(driver.reward_mean_Arr))
            np.save('reward_stds_ip.npy',np.array(driver.reward_std_Arr))
            
            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1


if __name__ == '__main__':
    # load environment
    #env = Environment(os.path.curdir, 'Hopper-v1')
    env = Environment(os.path.curdir, 'HalfCheetah-v1')
    env = Environment(os.path.curdir, 'InvertedPendulum-v1')

    # start training
    dispatcher(env=env)
