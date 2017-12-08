from mujoco_inverted_pendulum import MujocoInvPend

class PDController(MujocoInvPend):
    def __init__(self, env):
        MujocoInvPend.__init__(self, env)

    def act(self, observation):
        # define coefficients
        p_kp = 4
        p_kd = 1
        c_kp = 0.1
        c_kd = 0.2

        # calculate errors
        x = observation[0]
        theta = observation[1]
        v = observation[2]
        omega = observation[3]

        # combined PD controllers
        action = p_kp * theta + p_kd * omega + c_kp * x + c_kd * v

        # bound actions to within action space
        if action > self.env.action_space.high:
            action = self.env.action_space.high
        elif action < self.env.action_space.low:
            action = self.env.action_space.low
        self.action = action
        return action

    def update(self, obs, reward):
        return

    def step(self, action):
        return self.env.step(action)
