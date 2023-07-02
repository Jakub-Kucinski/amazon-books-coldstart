def process(mode):
    f = open("data/08_reporting/processed_results_on_" + mode + ".txt", "w")
    index = 0
    for line in open("data/08_reporting/results_on_" + mode + ".txt").read().split('\n'):

        if index < 3:
            f.write(line)
            f.write('\n')
            index += 1
            continue
        index += 1
        line = line.split('&')
        if len(line) < 3:
            continue
        while line[0][-1] == ' ':
            line[0] = line[0][:-1]
        f.write(line[0])
        for i in range(1, len(line)):
            value = 100 * float(line[i])
            f.write(f" | {value:.0f}")
        f.write('\n')

process("validation")
process("test")