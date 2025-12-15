import numpy as np
from algorithm import fft_recursive, fft_loop


if __name__ == "__main__":
    print(fft_recursive(np.array([1, 2, 3, 4])))
    print(fft_loop(np.array([1, 2, 3, 4])))
