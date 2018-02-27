import datetime

from dateutil import rrule


def get_minute_floor_datetime(dt):
    dt -= datetime.timedelta(seconds=dt.second,
                             microseconds=dt.microsecond)
    return dt


def minute_range(start_dt, end_dt, include_start=True, include_end=True):
    if include_start:
        yield start_dt

    rule = rrule.rrule(rrule.MINUTELY, bysecond=0, dtstart=start_dt)
    for x in rule.between(start_dt, end_dt, inc=False):
        yield x

    if include_end:
        yield end_dt


def custom_minute_range(start_dt, end_dt, interval, include_start=True, include_end=True):
    if include_start:
        yield start_dt

    rule = rrule.rrule(rrule.MINUTELY, interval=interval, bysecond=0, dtstart=start_dt)
    for x in rule.between(start_dt, end_dt, inc=False):
        yield x

    if include_end:
        yield end_dt


def hour_range(start_dt, end_dt, include_start=True, include_end=True):
    if include_start:
        yield start_dt

    rule = rrule.rrule(rrule.HOURLY, byminute=0, bysecond=0, dtstart=start_dt)
    for x in rule.between(start_dt, end_dt, inc=False):
        yield x

    if include_end:
        yield end_dt

def day_range(start_dt, end_dt, include_start=True, include_end=True):
    if include_start:
        yield start_dt

    rule = rrule.rrule(rrule.DAILY, byhour=0, byminute=0, bysecond=0, dtstart=start_dt)
    for x in rule.between(start_dt, end_dt, inc=False):
        yield x

    if include_end:
        yield end_dt
