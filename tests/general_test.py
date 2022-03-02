global_var = 0


def func_1():
    global global_var
    global_var = 1


def func_2():
    print(global_var)


if __name__ == '__main__':
    func_1()
    func_2()
