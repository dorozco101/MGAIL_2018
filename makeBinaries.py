import numpy as np
import cPickle
from ER import ER
import sys
env_name = sys.argv[1]
print(env_name)
sentiments = ['good','bad','mixed']
array_type = ['actions','nextStates','rewards','terminals','qpos','qvel']

for sentiment in sentiments:
    dump_path = 'expert_bins/'+sentiment+'_'+env_name+'_er.bin'
    paths = {}
    arrays = {}
    for Type in array_type:
        paths[Type] = 'expert_numpys/'+env_name+'_'+sentiment+'_'+Type+'.npy'
        arrays[Type] = np.load(paths[Type])
    
    arrays['nextStates'] = arrays['nextStates']
    action_size = arrays['actions'].shape[1]
    state_size = arrays['nextStates'].shape[1]
    qpos_size = arrays['qpos'].shape[1]
    qvel_size = arrays['qvel'].shape[1]
    print(action_size)
    print(state_size)
    print(qpos_size)
    print(qvel_size)
    er = ER(50000, state_size, action_size, 1, qpos_size, qvel_size, 70)
    er.add(arrays['actions'],arrays['rewards'],arrays['nextStates'],arrays['terminals'],arrays['qpos'],arrays['qvel'])
    cPickle.dump(er,open(dump_path,'wb'))