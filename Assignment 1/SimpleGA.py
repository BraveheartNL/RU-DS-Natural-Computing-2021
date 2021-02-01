import numpy as np
import matplotlib.pyplot as plt

def SimpleGA(l=100, iterations=1500):
    p = 1 / l
    # a) randomly generate a bit sequence:
    x = [np.random.randint(0, 1) for _ in range(l)]
    memory = [np.sum(x)]
    while iterations > 0:
        # b) create a copy of x and invert each of its bits with probability p. Let x_m be the result.
        x_m = [int(not bit) if np.random.rand() < p else bit for bit in x]
        # c) if x_m is closer to the goal sequence than x then replace x with x_m.
        # if np.sum(x_m) > np.sum(x):
        x = x_m
        memory.append(np.sum(x))
        iterations -= 1

    print("number of ones in found solution: {}".format(np.sum(x)))

    fig = plt.figure()
    plt.xlabel("Number of Iterations")
    plt.ylabel("Number of Ones")
    plt.plot(range(0, len(memory)), memory)
    fig.savefig("TheCountingOnes_SimpleGA_{}it_{}bits.jpg".format(len(memory), l))
    plt.show()


def main():
    SimpleGA()


if __name__ == "__main__":
    main()
