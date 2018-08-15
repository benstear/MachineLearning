#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:32:16 2018

@author: dawnstear
"""
'''
Terminal command for bulk download: $ curl "https://portals.broadinstitute.org/single_cell/bulk_data/oligodendroglioma-intra-tumor-heterogeneity/all/577300" -o cfg.txt; curl -K cfg.txt
'''

import pandas as pd

data = pd.read_table('/Users/dawnstear/desktop/chop_cellpred/cancer/OG_processed_data_portal.txt')
