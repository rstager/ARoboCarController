import pickle
import os
import tempfile
import importlib.util
import gym
from gym import spaces
import numpy as np
#controller side functions



# This class an aigym connection to the RoboCar simulator

class ARoboCar(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,config:{}):
        #print("ARoboCar config={}".format(config))
        observermodule=config['observer']
        self.config=config
        self.config.update(self._connect(config))

        observations=[]
        observations.append(spaces.Box(low=0, high=255, shape=(config['cameraheight'], config['camerawidth'], 3)))
        observations.append(spaces.Box(np.array([-1.0,-1.0,0.0]), np.array([1.0,1.0,100000.0]))) # speed, accel, odometer
        #if('observer' in self.config):
        #    observations.append(self.spaces(config))
        print(observations)
        self.observation_space = spaces.Tuple(observations)

        self.action_space = spaces.Box(np.array([-1.0,-1.0,0.0]), np.array([1.0,1.0,1.0])) # steer, gas, brake

        #self.reward_range((-1.0,1.0))


    def _step(self,action):
        command={"steering": action[0], "throttle": action[1]}
        # send command
        try:
            pickle.dump(command, self.fcmd)
            self.fcmd.flush()
            self.state= pickle.load(self.fstate)
            observation=self.state['observation']
            step_reward=self.state["reward"]
            done=self.state['done']
            info=self.state['info']
        except EOFError:
            print("Connection closed")
            self._disconnect()
            done=True
            step_reward = -1
            info = {}
            observation=None

        return observation, step_reward, done, info

    def _reset(self):
        for i in range(10):
            self.step([0,0,0])
        self.command({"command": "reset"})
        for i in range(10):
            self.step([0,0,0])
        ret,_,_,_ =  self.step([0,0,0])
        return ret

    def _close(self):
        self._disconnect()

    #def _seed(self):


    #def _render(self, mode='human', close=False):


    # argument can be a dictionary of config changes, or a callable that takes and returns a config dictionary.
    def _connect(self,confighook=None):
        tmpdir=tempfile.gettempdir()
        state_filename=os.path.join(tmpdir,"sim_state")
        cmd_filename=os.path.join(tmpdir,"sim_cmd")
        if not os.path.exists(state_filename):
            os.mkfifo(state_filename)
        if not os.path.exists(cmd_filename):
            os.mkfifo(cmd_filename)
        print("Connecting to server")
        self.fstate=open(state_filename,"rb")
        self.fcmd=open(cmd_filename,"wb")
        print("Connection opened")
        config = pickle.load(self.fstate)
        print("Got config={}".format(config))
        if confighook != None:
            if(callable(confighook)):
                config=confighook(config)
            else:
                config.update(confighook)
        if('observer' in config):
            with open(config['observer']+'.py', "r") as myfile:
                config['observercode'] = myfile.read()
#            config['observer']=importlib.util.find_spec(config['observer']).origin
        pickle.dump(config,self.fcmd)
        self.fcmd.flush()
        self.connected=True
        return config


    def _disconnect(self):
        self.fstate.close()
        self.fcmd.close()
        self.connected=False

    def command(self,command):
        try:
            pickle.dump(command, self.fcmd)
            self.fcmd.flush()
            self.state= pickle.load(self.fstate)
            return self.state
        except EOFError:
            print("Connection closed")
            self._disconnect()
            return None

