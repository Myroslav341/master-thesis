from time import time

import arrow

from queue_custom import Queue
from schema import ProtectedParam
from thread.thread_base import PeriodicThread


class QueryStatisticThread(PeriodicThread):
    def __init__(self, *args, query: Queue, field_to_update: ProtectedParam, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = query

        self.item_count_over_time = dict()

        self.start_time = None
        self.field_to_update = field_to_update

    def handler(self):
        if not self.start_time:
            self.start_time = time()

        item_count = self.query.get_item_count()

        self.field_to_update.set(item_count)

        timedelta = (time() - self.start_time)

        self.item_count_over_time[timedelta] = item_count
