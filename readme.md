# weather prediction and score

## how to run

```python
# default:
year = 2020
month = 1
day = 25
hour = 8

immediate = False		# 是否立即出发
nightmode = False		# 夜晚是否想游玩

# 只有 immediate 为 False 时才生效
start_year = 2020
start_month = 1
start_day = 26

end_year = 2020
end_month = 1
end_day = 27
```



- 实时，求得今日游玩推荐

```python
python run.py --year 2020 --month 1 --day 25 --hour 8 --immediate --nightmode
```

- 未来几天游玩推荐,start dat to end day

```python
python run.py --start_year 2020 --start_month 1 --start_day 26 --end_year 2020 --end_month 1 --end_day 27
```

