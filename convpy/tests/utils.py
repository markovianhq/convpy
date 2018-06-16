import os
import re


def parse_no_events():
    no_events = os.getenv('CONVPY_TEST_NO_EVENTS', (1e4,))

    if isinstance(no_events, str):
        no_events = re.sub(r'\s+', r'', no_events)
        no_events = re.sub(r'\(', r'', no_events)
        no_events = re.sub(r'\)', r'', no_events)
        no_events = no_events.split(',')

        no_events = tuple(int(el) for el in no_events if len(el.strip()) > 0)

    return no_events


def parse_no_runs():
    no_runs = os.getenv('CONVPY_TEST_NO_RUNS', 1)

    if isinstance(no_runs, str):
        no_runs = re.match(r'(\d+)', no_runs).groups()[0]

        no_runs = int(no_runs)

    return no_runs
