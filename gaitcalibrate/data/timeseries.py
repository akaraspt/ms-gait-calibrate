
class Acceleration(object):

    def __init__(self,
                 data,
                 fs):
        self.data = data
        self.fs = fs

        # Nyquist frequency
        self.fnyq = self.fs * 0.5

        # Number of samples
        self.size = len(self.data)

        # Duration in seconds
        self.duration = len(self.data) / float(self.fs)

    def get_idx(self, idx):
        return Acceleration(data=self.data.iloc[idx],
                            fs=self.fs)

    def get_dt(self,
               start_dt,
               end_dt):
        return Acceleration(data=self.data[((self.data['dt'] >= start_dt) &
                                            (self.data['dt'] <= end_dt))],
                            fs=self.fs)


class WalkSpeed(object):

    def __init__(self,
                 data):
        self.data = data

        # Number of samples
        self.size = len(self.data)

    def get_idx(self, idx):
        return WalkSpeed(data=self.data.iloc[idx])

    def get_dt(self,
               start_dt,
               end_dt):
        return WalkSpeed(data=self.data[((self.data['dt'] >= start_dt) &
                                         (self.data['dt'] <= end_dt))])
