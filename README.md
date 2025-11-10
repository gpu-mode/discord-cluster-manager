# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/discord-cluster-manager/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py                         |       80 |        9 |     89% |38-39, 59, 200-202, 219-221 |
| src/libkernelbot/background\_submission\_manager.py |      160 |       31 |     81% |36, 38-40, 42, 45, 47, 176-177, 203-206, 224-229, 246-271 |
| src/libkernelbot/consts.py                          |       65 |        1 |     98% |        48 |
| src/libkernelbot/db\_types.py                       |       14 |        1 |     93% |         7 |
| src/libkernelbot/leaderboard\_db.py                 |      289 |       48 |     83% |65, 99, 373-383, 396-414, 719-721, 790-811, 968-992, 1004-1043, 1050-1071, 1078-1085, 1101-1110 |
| src/libkernelbot/report.py                          |      267 |        8 |     97% |73, 339, 350, 388, 415, 422-423, 430 |
| src/libkernelbot/submission.py                      |      121 |        1 |     99% |        18 |
| src/libkernelbot/task.py                            |      110 |        6 |     95% |68, 119, 124-126, 163 |
| src/libkernelbot/utils.py                           |       87 |        5 |     94% |     39-48 |
|                                           **TOTAL** | **1193** |  **110** | **91%** |           |


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