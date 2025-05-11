import time
import threading

class RateLimiter:
    def __init__(self, max_calls, period=1):
        """
        :param max_calls: Maximum number of calls allowed in a period.
        :param period: Period (in seconds) over which the calls are counted.
        """
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []  # timestamps of recent calls

    def wait(self):
        with self.lock:
            now = time.time()
            # Remove timestamps older than the period.
            self.calls = [timestamp for timestamp in self.calls if now - timestamp < self.period]

            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait until the oldest call is outside the period.
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Update the current time and cleanup the calls again.
                now = time.time()
                self.calls = [timestamp for timestamp in self.calls if now - timestamp < self.period]

            # Record the current call.
            self.calls.append(time.time())