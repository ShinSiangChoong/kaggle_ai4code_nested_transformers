
def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()
