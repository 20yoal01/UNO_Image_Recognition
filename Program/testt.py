import numpy as np

blue_responses = np.zeros(10, dtype=np.float32)
red_resposnes = np.ones(10, dtype=np.float32)
responses = np.concatenate((blue_responses, red_resposnes))

print(responses)