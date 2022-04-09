import tensorflow as tf
from train_model import Train

one_step_model = Train().get_one_step()
states = None
next_char = tf.constant(['Shrek:'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)