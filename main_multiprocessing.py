import multiprocessing 
import os
from main import run
from src.brc import BRC
import datetime
import pdb

if __name__ == '__main__':

    # Blacklist existing output
    existing_files = os.listdir('pricingkernel/plots')

    # Make sure path exists
    bandwidths = [0.08]#[0.02, 0.04, 0.06, 0.08]
    for bw in bandwidths:
        pth = 'pricingkernel20220222/plots/' + str(bw)
        if not os.path.exists(pth):
            os.makedirs(pth)

    brc = BRC()
    #curr_date = brc.first_day
    curr_date = datetime.datetime(2020, 3, 4,0,0,0)
    run_dates = [curr_date]
    while curr_date < brc.last_day:
        curr_date += datetime.timedelta(1)
        out = next((s for s in existing_files if curr_date.strftime('%Y-%m-%d') in s), None) 
        print(out)
        if out is None:
            run_dates.append(curr_date)

    multiprocessing.set_start_method('spawn')
    n_cpu = min(multiprocessing.cpu_count(), 4)
    p = multiprocessing.Pool(processes = n_cpu)
    p.map(run, run_dates)
