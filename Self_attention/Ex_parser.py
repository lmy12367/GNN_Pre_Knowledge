import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="excrise paraser")

    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    return args


class IOStream:
    def __init__(self, file_name):
        self.file = open(file_name, "a")

    def cprint(self, text):
        msg = f"[{time.strftime('%H:%M:%S')}] {text}"

        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

def table_printer(args):
    args=vars(args)
    print("*"*30)
    print("parameters     | val")
    print("*"*30)
    for k,v in sorted(args.items()):
        print(f"|{k:<13} | {v:>5}")

    print(("*"*30))

if __name__ == "__main__":
    args = parse_args()
    log = IOStream("Ex_log.txt")
    log.cprint("Hyper-parameters:")
    log.cprint(table_printer(args))
    log.close()

