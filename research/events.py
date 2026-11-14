"""Scheduled-event calendar used for research: FOMC decision days (corpora list + well-known
scheduled dates filling the gaps) for years with a complete calendar."""
import subprocess
import pandas as pd
_c = subprocess.run("ls ../track2-forecasting-public/units/*/text/ | grep -oE 'fomc_statement_[0-9]{8}' | sort -u | sed 's/fomc_statement_//'",
                    shell=True, capture_output=True, text=True).stdout.split()
EXTRA = ['20030916', '20160316', '20160727', '20161102', '20161214', '20170201', '20170503', '20170920', '20171101',
         '20180131', '20180502', '20180801', '20190320', '20191211', '20210127', '20210317', '20210428', '20210728',
         '20221102', '20221214', '20230322', '20230503', '20240320']
FOMC = sorted(set(pd.to_datetime(_c + EXTRA)))
COMPLETE = [2003, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
