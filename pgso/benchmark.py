from multiprocessing import Process
import psutil
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def monitor(target, bounds, num_particles, max_iter, costfunc, M):
    worker_process = Process(target=target, args=(M, bounds, num_particles, max_iter, costfunc))
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = [0.0]
    start = time.time()
    time_at = []
    top_prcnt = []
    while worker_process.is_alive():
        top = psutil.cpu_percent(percpu=True)
        top_prcnt.append(top)
        cpu_percents.append(p.cpu_percent())
        time_at.append(time.time()-start)
        time.sleep(0.01)
    worker_process.join()
    return cpu_percents, time_at, top_prcnt

def multiple_cpu_plot(top_prcnt, time_at, zoom_range=[], step=1):
    cols = list()
    for i in range(psutil.cpu_count()):
        cols.append('cpu_'+str(i+1))
    df = pd.DataFrame.from_records(top_prcnt, columns=cols)
    df['time_at'] = time_at   
    fig, ax = plt.subplots(figsize=(20,20), ncols=3, nrows=3)

    sns.set_style("dark")
    flat_ax = [bx for axs in ax for bx in axs]
    for i,sdf in enumerate(flat_ax):
        if i >= psutil.cpu_count():
            break
        if zoom_range!= []:
            sdf.set_xticks((np.arange(zoom_range[0], zoom_range[1]+1, step)))
            sdf.set_xlim(zoom_range[0], zoom_range[1])
        sdf.set_title('cpu_'+str(i+1))
    for i in range(psutil.cpu_count()):
        sns.lineplot(x='time_at', y='cpu_'+str(i+1), data=df, ax=flat_ax[i])