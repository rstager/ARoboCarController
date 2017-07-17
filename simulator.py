import pickle
import os
import tempfile
import importlib
# This class manages a connection to the RoboCar simulator

class Simulator:
    # argument can be a dictionary of config changes, or a callable that takes and returns a config dictionary.
    def connect(self,confighook=None):
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
        if confighook != None:
            if(callable(confighook)):
                config=confighook(config)
            else:
                config.update(confighook)
        if('controller' in config):
            config['controller']=importlib.util.find_spec(config['observer']).origin
        pickle.dump(config,self.fcmd)
        self.fcmd.flush()
        return config

    def get_state(self):
        try:
            return pickle.load(self.fstate)
        except EOFError:
            print("Connection closed")
            return None

    def send_cmd(self,command):
        try:
            pickle.dump(command, self.fcmd)
            self.fcmd.flush()
        except EOFError:
            print("Connection closed")
            self.disconnect()
            return None

    def disconnect(self):
        self.fstate.close()
        self.fcmd.close()

    def reset(self):
        for i in range(10):
            tmp=self.get_state()
            self.send_cmd({"steering": 0, "throttle": 0})
        print("send reset")
        tmp = self.get_state()
        self.send_cmd({"command": "reset"})
        for i in range(10):
            tmp=self.get_state()
            self.send_cmd({"steering": 0, "throttle": 0})