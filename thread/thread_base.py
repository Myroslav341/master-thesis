from threading import Thread
from time import sleep

from internal_types import IsRunningGetter


class PeriodicThread(Thread):
    def __init__(self, is_running_getter: IsRunningGetter, refresh_time: int):
        Thread.__init__(self)
        self.refresh_time = refresh_time
        self.is_running_getter = is_running_getter

    def run(self):
        while True:
            if self.is_running_getter() is False:
                break
            self.handler()
            if self.is_running_getter() is False:
                break
            sleep(self.refresh_time)

    def handler(self):
        raise NotImplementedError()


class InfiniteThread(Thread):
    def __init__(self, is_running_getter: IsRunningGetter, *args, **kwargs):
        Thread.__init__(self)
        self.is_running_getter = is_running_getter

    def run(self):
        self.handler()

    def handler(self):
        raise NotImplementedError()

