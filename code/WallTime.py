import time

class WallTimeWatcher():
    '''Return boolean indicating whether walltime might be reached in the next episode
    '''
    def __init__(self, start_time, walltime):
        self.start_time = start_time
        self.walltime = walltime
        self.wait = 0
        self.episode_start = 0
        self.episode_average = 0
        self.reachedWalltime = False

    def on_episode_begin(self):
        self.episode_start=time.time()

    def on_episode_end(self):
        episodetime = (time.time() - self.episode_start)
        # update average completion time
        if self.episode_average > 0:
            self.episode_average = (self.episode_average + episodetime) / 2
        else:
            self.episode_average = episodetime
        # 
        if (time.time() - self.start_time + 3*self.episode_average)>self.walltime:
            print("will run over walltime: %s s" % int(time.time() - self.start_time + 3*self.episode_average))
            self.reachedWalltime = True
            return self.reachedWalltime

