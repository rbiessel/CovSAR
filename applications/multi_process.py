from multiprocessing import Process


def f(name):
    for i in range(5):
        print(i)


if __name__ == '__main__':

    for i in range(10):
        p = Process(target=f, args=('bob',))
        p.start()
