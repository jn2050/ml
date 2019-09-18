import re


class Environment():
    def __init__(self):
        re1 = re.compile(r'''^([^\s=]+)=(?:[\s"']*)(.+?)(?:[\s"']*)$''')
        self.ENV = {}
        with open('.env') as f:
            for line in f:
                match = re1.match(line)
                if match is not None:
                    self.ENV[match.group(1)] = match.group(2)

    def get(self, var):
        return self.ENV[var]

ENV = Environment()
