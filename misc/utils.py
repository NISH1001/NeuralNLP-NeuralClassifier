#!/usr/bin/env python3


import calendar
import datetime
import os
import time


class IAmTime:
    def __init__(self):
        now = datetime.datetime.now()
        month = now.strftime("%m")
        day = now.strftime("%d")
        self.month = str(month.lower())
        self.year = now.year
        self.day = day
        self.hour = now.hour
        self.minute = now.minute
        self.second = now.second

    def __repr__(self) -> str:
        return f"(year={self.year}, month={self.month}, day={self.day}, hour={self.hour}, minute={self.minute}, second={self.second})"


def create_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    pass


if __name__ == "__main__":
    main()
