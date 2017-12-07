class PDController(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, initial, observation, reward, done):
        # define coefficients
        p_kp = 4
        p_kd = 1
        c_kp = 0.1
        c_kd = 0.2

        # calculate errors
        x = observation[0] - initial[0]
        theta = observation[1] - initial[1]
        v = observation[2] - initial[2]
        omega = observation[3] - initial[3]

        # combined PD controllers
        action = p_kp * theta + p_kd * omega + c_kp * x + c_kd * v

        # bound actions to within action space
        if action > self.action_space.high:
            action = self.action_space.high
        elif action < self.action_space.low:
            action = self.action_space.low
        return action
