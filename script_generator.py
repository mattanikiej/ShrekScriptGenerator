import tensorflow as tf
from models.train_model import Train


class ScriptGenerator():
    """
    Class that is used to generate the scripts
    """
    def __init__(self):
        self.one_step_model = Train(load=True, name='V1').get_one_step()
        self.script_title = 'Shrek And The Fool Who Can\'t Read Directions'
        self.first_words = "Get out me swamp!"

    def generate_script(self):
        """
        Generates the custom script from the user's input
        :return: the generated script
        """

        # reset states
        states = None

        # build initial data
        initial_data = self.script_title + '\nSHREK:\n' + self.first_words
        next_char = tf.constant([initial_data])
        result = [next_char]

        # generate 3000 characters
        for n in range(3000):
            next_char, states = self.one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        # join characters
        result = tf.strings.join(result)
        return result[0].numpy().decode('utf-8') + '\n\n' + '_' * 80 + '\n'

    def set_script_title(self, script_title):
        """
        Setter for self.script_title
        """
        self.script_title = script_title

    def set_first_words(self, first_words):
        """
        Setter for self.script_title
        """
        self.first_words = first_words
