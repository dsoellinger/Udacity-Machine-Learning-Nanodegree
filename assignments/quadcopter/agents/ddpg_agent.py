from actors import SmartActor


class DDPGAgent():

    def __init__(self, task):

        self.state_size = task.state_size
        self.action_size = task.action_size

        self.actor = SmartActor(self.state_size, self.action_size)
        self.task = task

    def reset_episode(self):
        state = self.task.reset()
        return state

    def act(self, state):
        action = self.actor.predict_action(state)
        return action
