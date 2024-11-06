from timeit import default_timer as timer


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(f"** {func.__name__} executed in {end - start} seconds")
        return result
    return wrapper