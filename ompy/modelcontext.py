import inspect


class ModelContext:
    def __init__(self):
        self._defaults = {'x': 5, 'a': 8}
        for attr, val in self._defaults.items():
            setattr(self, attr, val)

    def __enter__(self):
        # print("IN ENTER")
        # stack = inspect.stack()
        # print(stack[1][0].f_locals)
        # print("LEAVE ENTER")
        # frame = inspect.currentframe().f_back
        # self.local_vars = frame.f_locals
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr, val in self._defaults.items():
            if getattr(self, attr) != val:
                print(f"{attr} <> {getattr(self, attr)}")
            else:
                print(f"{attr} = {val}")
        return

        # Can't find a way for the pretty notation
        # frame = inspect.currentframe().f_back
        # local_vars = frame.f_locals
        # print(local_vars)
        # names = frame.f_code.co_names
        # print("====")
        # print(names)
        # frame = inspect.currentframe().f_back
        # local_vars = frame.f_locals
        # print("<<<<<")
        # print(local_vars)
        # print("====")
        # print(self.local_vars)
        # print(">>>>>")
        # stack = inspect.stack()
        # print(stack[1][0].f_locals)
        # print("<><><><>")

        #for (name, val) in local_vars.items():
        #    print(name, val)
        #    if name in self.values:
        #        self.values[name] = local_vars[name]

        #for key, val in self.values.items():
        #    if self.defaults[key] == val:
        #        print(f"Key {key} is unchanged")

