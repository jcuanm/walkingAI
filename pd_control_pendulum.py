class PDController(object):
    def __init__(self, env):
        self.env = env

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
        return action

    def update(self, obs, reward):
        return

    def get_explore_rate(self, episode):
        return
    def get_learning_rate(self, episode):
        return
