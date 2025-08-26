# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/discord-cluster-manager/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py         |       78 |        8 |     90% |38-39, 195-197, 214-216 |
| src/libkernelbot/consts.py          |       66 |        1 |     98% |        45 |
| src/libkernelbot/db\_types.py       |       47 |        1 |     98% |         6 |
| src/libkernelbot/leaderboard\_db.py |      273 |       40 |     85% |59, 94, 626-628, 697-718, 875-899, 911-950, 957-978, 985-992, 1008-1017 |
| src/libkernelbot/report.py          |      252 |        8 |     97% |46, 308, 318, 339, 366, 373-374, 381 |
| src/libkernelbot/submission.py      |      130 |        1 |     99% |        18 |
| src/libkernelbot/task.py            |      106 |        5 |     95% |67, 117, 122-124 |
| src/libkernelbot/utils.py           |       87 |        5 |     94% |     39-48 |
|                           **TOTAL** | **1039** |   **69** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/gpu-mode/discord-cluster-manager/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/gpu-mode/discord-cluster-manager/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/gpu-mode/discord-cluster-manager/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/gpu-mode/discord-cluster-manager/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgpu-mode%2Fdiscord-cluster-manager%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/gpu-mode/discord-cluster-manager/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.