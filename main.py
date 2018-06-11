import numpy as np
import os
from environment import Environment
from driver import Driver

import matplotlib.pyplot as plt


def save_fig(itrs,means,stds, filepath="./graph_rewards.png",title = 'rewards vs itr',
             x_label="iteration", y_label="avg reward", x_range=(0, 1), y_range=(0,1), color="blue",  grid=True):
  fig = plt.figure()
  #ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
  ax = fig.add_subplot(111, autoscale_on=True, xlim=x_range, ylim=y_range)
  ax.grid(grid)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.plot(itrs,means, color,  alpha=1.0)
  ax.fill_between(itrs,means+stds, means-stds, facecolor=color, alpha=0.5)
  fig.savefig(filepath)
  fig.clear()
  plt.close(fig)
  
def dispatcher(env, ER_name,end_condition, test_interval, n_episodes_test,min_iter_test):

    driver = Driver(env,ER_name)
    driver_itrs = []


    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if (driver.itr % test_interval == 0) and (driver.itr >= min_iter_test):

            # measure performance
            R = []
            for n in range(n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=False, noise_flag=False, n_steps=1000))
            #vis=env.vis_flag
            # update stats
            driver.reward_mean = float(sum(R)) / len(R)
            driver.reward_std = np.std(R)
            
            #save running rewards and stds
            #driver_itrs.append(driver.itr)
            #driver.reward_mean_Arr.append(driver.reward_mean)
            #driver.reward_std_Arr.append(driver.reward_std)
            
            #commented out so less popups 
            #save_fig(np.array(driver_itrs),np.array(driver.reward_mean_Arr),np.array(driver.reward_std_Arr) ,'results/rewards_plot_'+ER_name+'.png', my_ER_name,'iteration','avg_reward')
            
            #np.savetxt('results/reward_itrs_'+ER_name+'.csv', np.array(driver_itrs), delimiter=",")
            #np.savetxt('results/reward_means_'+ER_name+'.csv', np.array(driver.reward_mean_Arr), delimiter=",")
            #np.savetxt('results/reward_stds_'+ER_name+'.csv', np.array(driver.reward_std_Arr), delimiter=",")
            
            #np.save('reward_itrs_hop.npy',np.array(driver_itrs))
            #np.save('reward_means_hop.npy',np.array(driver.reward_mean_Arr))
            #np.save('reward_stds_hop.npy',np.array(driver.reward_std_Arr))
            
            # print info line
            driver.print_info_line('full')

            # save snapshot
            #if env.train_mode and env.save_models:
            #   driver.save_model(dir_name=env.config_dir)
                
        if driver.reward_mean >= end_condition:
            print(driver.reward_mean),
            print(' '),
            print(driver.itr)
            """
            plt.figure()
            plt.plot(R)
            plt.title('MGAIL,' + ER_name +' itr = ' + str(driver.itr))
            plt.show()
            save_fig(np.array(driver_itrs),np.array(driver.reward_mean_Arr),np.array(driver.reward_std_Arr) ,'results/rewards_plot_'+ER_name+'.png', 'MGAIL: '+ my_ER_name,'iteration','avg_reward')
            """
            break

        driver.itr += 1


if __name__ == '__main__':
    # load environment
    my_env_name = 'InvertedPendulum-v1'
    my_ER_name = 'mixed_'+my_env_name+'_er'
    er_size = 50000
    env = Environment(os.path.curdir, my_env_name, my_ER_name,er_size)
    if my_env_name == 'InvertedPendulum-v1':
        test_interval = 100
        term_condition = 990
        n_episodes_test = 5
        min_iter_test = 1000

    elif my_env_name == 'HalfCheetah-v1':
        term_condition = 2000
        test_interval = 400
        n_episodes_test = 5
    else:
        term_condition = 2500
        test_interval = 400  
        n_episodes_test = 5     

    # start training
    dispatcher(env=env,ER_name = my_ER_name,end_condition = term_condition, test_interval = test_interval,n_episodes_test = n_episodes_test, min_iter_test = min_iter_test)
