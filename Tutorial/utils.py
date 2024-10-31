import warnings
warnings.simplefilter("ignore")
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime, read, read_inventory
from obspy.signal.rotate import rotate2zne, rotate_ne_rt
from obspy.signal.cross_correlation import correlate
import matplotlib.pyplot as plt
import matplotlib as mpl


class BazAnalysis:
    def __init__(self, inventory_file, data_file, 
                 filter_freq=(3, 15),           # (low, high) in Hz
                 time_window=(0, 180),          # (start, end) in seconds
                 window_params=(1.0, 0.02),     # (length, step) in seconds
                 station_pairs=None,            # List of (accelerometer, rotational) pairs
                 cc_threshold=0.4):             # Cross-correlation coefficient threshold
        """
        Initialize SeismicAnalysis with configurable parameters.
        
        Parameters:
        -----------
        inventory_file : str
            Path to the inventory XML file
        data_file : str
            Path to the waveform data file
        filter_freq : tuple (float, float)
            Low and high frequency bounds for bandpass filter in Hz
        time_window : tuple (float, float)
            Start and end times for analysis in seconds
        window_params : tuple (float, float)
            Length and step size of the moving window in seconds
        station_pairs : list of tuples, optional
            List of (accelerometer, rotational) station pairs
            Default: [('TCBS3', 'BS3'), ('TCBS4', 'BS4')]
        cc_threshold : float
            Cross-correlation coefficient threshold for visualization
        """
        # Set default station pairs if none provided
        if station_pairs is None:
            station_pairs = [('TCBS3', 'BS3'), ('TCBS4', 'BS4')]
            
        # Configuration parameters
        self.filter_freq = {
            'low': float(filter_freq[0]),
            'high': float(filter_freq[1])
        }
        
        self.time_window = {
            'start': float(time_window[0]),
            'end': float(time_window[1])
        }
        
        self.window_params = {
            'length': float(window_params[0]),
            'step': float(window_params[1])
        }
        
        # Separate station pairs into two lists
        self.stations = {
            'accelerometers': [pair[0] for pair in station_pairs],
            'rotational': [pair[1] for pair in station_pairs]
        }
        
        self.cc_threshold = float(cc_threshold)
        
        # Validate inputs
        self._validate_inputs()
        
        # Load data
        self.inventory = read_inventory(inventory_file)
        self.waveform = read(data_file)
        
        # Initialize processing parameters
        self.dt = self.waveform[0].stats.delta
        self.setup_time_indices()
    
    def _validate_inputs(self):
        """Validate input parameters"""
        # Frequency validation
        if self.filter_freq['low'] <= 0 or self.filter_freq['high'] <= 0:
            raise ValueError("Filter frequencies must be positive")
        if self.filter_freq['low'] >= self.filter_freq['high']:
            raise ValueError("High frequency must be greater than low frequency")
        
        # Time window validation
        if self.time_window['start'] < 0 or self.time_window['end'] <= 0:
            raise ValueError("Time window values must be non-negative")
        if self.time_window['start'] >= self.time_window['end']:
            raise ValueError("End time must be greater than start time")
        
        # Window parameters validation
        if self.window_params['length'] <= 0 or self.window_params['step'] <= 0:
            raise ValueError("Window length and step must be positive")
        if self.window_params['step'] >= self.window_params['length']:
            raise ValueError("Window step must be smaller than window length")
        
        # Station pairs validation
        if len(self.stations['accelerometers']) != len(self.stations['rotational']):
            raise ValueError("Number of accelerometer and rotational stations must match")
        if not self.stations['accelerometers'] or not self.stations['rotational']:
            raise ValueError("At least one station pair must be provided")
        
        # Cross-correlation threshold validation
        if not 0 <= self.cc_threshold <= 1:
            raise ValueError("Cross-correlation threshold must be between 0 and 1")
        
    def setup_time_indices(self):
        """Setup time indices for processing windows"""
        t1, t2 = self.time_window['start'], self.time_window['end']
        self.t_indices = {
            'start': np.int32(t1 / self.dt),
            'end': np.int32(t2 / self.dt)
        }
        self.wins = np.arange(0, t2 - t1 + self.window_params['step'], 
                            self.window_params['step'])
        self.num_windows = len(self.wins)
        
        # Initialize results arrays
        self.results = {
            'baz': np.empty((self.num_windows, 3)),
            'corr': np.empty((self.num_windows, 2))
        }

    def plot_6c_seismogram(self, station='*BS4'):
        """
        Plot 6-component seismogram data showing translation and rotation measurements.
        
        Parameters:
        -----------
        waveform_fil : ObsPy Stream object
            Filtered waveform data containing 6 components
        station : str, optional
            Station identifier (default: '*BS4')
            Use '*BS3' for northern station or '*BS4' for southern station
        
        Returns:
        --------
        fig : matplotlib Figure
            The generated figure object
        axs : array of Axes
            Array of subplot axes objects
        """
        # Define channel components
        channels = {
            'acceleration': ['HHE', 'HHN', 'HHZ'],  # Acceleration components
            'rotation': ['HJE', 'HJN', 'HJZ']       # Rotation components
        }
        all_channels = channels['acceleration'] + channels['rotation']
        
        # Create figure and subplots
        fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.0)
        
        # Set title
        station_id = station[1:]  # Remove the '*' prefix
        fig.suptitle(f"6C - Traffic signals - Station TC{station_id}-{station_id}", 
                    fontsize=14, y=0.99)
        
        # Plot each component
        for idx, channel in enumerate(all_channels):
            # Get waveform data for current channel
            trace = self.waveform.select(station=station, channel=channel)[0]
            
            # Plot time series
            axs[idx].plot(trace.times(), 
                         trace.data,
                         color='k',
                         linewidth=0.5,
                         label=channel)
            
            if idx < 3:
                axs[idx].set_ylabel(r"Acc (m/s$^{2}$)")
            else:
                axs[idx].set_ylabel(r"Rot (rad/s)")
            
            # Add legend
            axs[idx].legend(loc=1)
        
        # Set x-axis label on bottom subplot
        axs[-1].set_xlabel("Time (s)")
        
        # Adjust subplot spacing
        plt.subplots_adjust(
            left=0.1,
            bottom=0.07,
            right=0.95,
            top=0.95,
            hspace=0.0
        )
    
        return fig, axs
    
    def process_station_data(self):
        """Process data for each station pair"""
        for num_sta, acc_sta in enumerate(self.stations['accelerometers']):
            rot_sta = self.stations['rotational'][num_sta]
            baz_cal, times, corrbaz = [], [], []
            
            for win_start in self.wins:
                window_results = self.analyze_time_window(num_sta, win_start)
                baz_cal.append(window_results['baz'])
                times.append(win_start + self.window_params['length'])
                corrbaz.append(window_results['corr'])
            
            # Store results
            self.results['baz'][:, num_sta + 1] = baz_cal
            self.results['corr'][:, num_sta] = corrbaz
            
            print(f"Processed station pair: {acc_sta} - {rot_sta}")
            
        self.results['baz'][:, 0] = times
    
    def analyze_time_window(self, station_idx, window_start):
        """Analyze a single time window for back azimuth estimation"""
        # Calculate window indices
        t1 = self.time_window['start'] + window_start
        t2 = t1 + self.window_params['length']
        idx1 = np.int32(t1 / self.dt)
        idx2 = np.int32(t2 / self.dt)
        
        # Get rotational component data
        rot_sta = self.stations['rotational'][station_idx]
        data = np.column_stack((
            self.waveform.select(station=rot_sta, channel='HJE')[0].data[idx1:idx2],
            self.waveform.select(station=rot_sta, channel='HJN')[0].data[idx1:idx2]
        ))
        
        # Perform polarization analysis
        baz = self.calculate_back_azimuth(data)
        
        # Resolve 180° ambiguity
        acc_sta = self.stations['accelerometers'][station_idx]
        baz,corr = self.resolve_ambiguity(acc_sta, rot_sta, baz, idx1, idx2)
        
        return {'baz': baz, 'corr': np.abs(corr[0])}
    
    def calculate_back_azimuth(self, data):
        """Calculate back azimuth using polarization analysis"""
        C = np.cov(data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(C, UPLO='U')
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        Q = eigenvectors[:, idx]
        
        baz = -np.arctan((Q[1, 0] / Q[0, 0])) * 180 / np.pi
        return baz + 180.0 if baz <= 0 else baz
    
    def resolve_ambiguity(self, acc_sta, rot_sta, baz, idx1, idx2):
        """Resolve 180° ambiguity using cross-correlation"""
        z_data = self.waveform.select(station=acc_sta, channel='HHZ')[0].data[idx1:idx2]
        
        rot_data = rotate_ne_rt(
            self.waveform.select(station=rot_sta, channel='HJN')[0].data[idx1:idx2],
            self.waveform.select(station=rot_sta, channel='HJE')[0].data[idx1:idx2],
            baz
        )[1][:]
        
        corr = correlate(z_data, rot_data, 0)
         
        return baz + 180.0 if corr[0] > 0 else baz, corr
    
    def plot_results(self, station_idx=1):
        """Create visualization of the results"""
        starttime = max([tr.stats.starttime for tr in self.waveform])
        endtime = starttime + self.time_window['end']
        
        fig = plt.figure(figsize=(12, 5))
        
        # Setup time axes and limits
        time_acc = self.waveform.select(
            station=self.stations['accelerometers'][station_idx], 
            channel='HHZ')[0].times()
        time_rot = self.waveform.select(
            station=self.stations['rotational'][station_idx], 
            channel='HJE')[0].times()
        
        self._plot_acceleration(fig, station_idx, time_acc, starttime, endtime)
        self._plot_rotation(fig, station_idx, time_rot)
        self._plot_back_azimuth(fig, station_idx)
        self._add_colorbar(fig)
        
        plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.92, hspace=0.0)
        plt.show()
    
    def _plot_acceleration(self, fig, station_idx, time, starttime, endtime):
        """Plot acceleration data"""
        ax = plt.subplot2grid((3, 50), (0, 0), colspan=49)
        data = self.waveform.select(
            station=self.stations['accelerometers'][station_idx], 
            channel='HHZ')[0].data
        
        ylim = np.max(np.abs(data))
        ax.plot(time, data, 'k', linewidth=0.5, label="BS-Az")
        
        self._format_acceleration_plot(ax, starttime, endtime, ylim)
    
    def _plot_rotation(self, fig, station_idx, time):
        """Plot rotational rate data"""
        ax = plt.subplot2grid((3, 50), (1, 0), colspan=49)
        rot_sta = self.stations['rotational'][station_idx]
        
        data_e = self.waveform.select(station=rot_sta, channel='HJE')[0].data
        data_n = self.waveform.select(station=rot_sta, channel='HJN')[0].data
        
        ylim = np.max([np.abs(data_e), np.abs(data_n)])
        
        self._format_rotation_plot(ax, time, data_e, data_n, ylim)
    
    def _plot_back_azimuth(self, fig, station_idx):
        """Plot back azimuth results"""
        ax = plt.subplot2grid((3, 50), (2, 0), colspan=49)
        
        mask = self.results['corr'][:, station_idx] > self.cc_threshold
        times = self.results['baz'][mask, 0]
        baz = self.results['baz'][mask, station_idx + 1]
        cc = self.results['corr'][mask, station_idx]
        
        self._format_baz_plot(ax, times, baz, cc)
    
    def _format_acceleration_plot(self, ax, starttime, endtime, ylim):
        """Format acceleration subplot"""
        t1, t2 = self.time_window['start'], self.time_window['end']
        ax.set_title(f'{starttime.datetime:%Y-%m-%d %H:%M} - {endtime.datetime:%H:%M}')
        ax.annotate(f"{self.filter_freq['low']} - {self.filter_freq['high']} Hz", 
                   xy=(t2 - 12, -ylim * 0.9), fontsize=10, color='r')
        ax.set_xlim(t1 + self.window_params['length'], t2 + self.window_params['length'])
        ax.set_ylim(-ylim, ylim)
        ax.set_xticklabels([])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_ylabel("Acceleration\n(m/s²)")
        ax.legend(loc=1, ncol=2, prop={'size': 10})
    
    def _format_rotation_plot(self, ax, time, data_e, data_n, ylim):
        """Format rotation subplot"""
        t1, t2 = self.time_window['start'], self.time_window['end']
        ax.plot(time, data_e, 'k', linewidth=0.5, label="BS-Re")
        ax.plot(time, data_n, 'r', linewidth=0.5, label="BS-Rn")
        ax.annotate(f"{self.filter_freq['low']} - {self.filter_freq['high']} Hz", 
                   xy=(t2 - 12, -ylim * 0.9), fontsize=10, color='r')
        ax.set_xlim(t1 + self.window_params['length'], t2 + self.window_params['length'])
        ax.set_ylim(-ylim, ylim)
        ax.set_xticklabels([])
        ax.set_ylabel("Rotational rate\n(rad/s)")
        ax.legend(loc=1, ncol=2, prop={'size': 10})
    
    def _format_baz_plot(self, ax, times, baz, cc):
        """Format back azimuth subplot"""
        t1, t2 = self.time_window['start'], self.time_window['end']
        ax.scatter(times, baz, c=cc, cmap=plt.cm.RdYlGn_r,
                  vmin=0.0, vmax=1.0, marker='.', s=55, alpha=0.7,
                  label='Estimated Baz from Re/Rn')
        ax.set_xlim(t1 + self.window_params['length'], t2 + self.window_params['length'])
        ax.set_ylim(0, 360)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Back azimuth\n(°)")
        ax.annotate(f'CC > {self.cc_threshold:.2f}', xy=(t2 - 12, 130), 
                   fontsize=10, color='r')
        ax.hlines(y=300, xmin=t1 + self.window_params['length'], 
                 xmax=t2 + self.window_params['length'],
                 linestyle=':', linewidth=2, color='royalblue', label='300°')
        ax.hlines(y=100, xmin=t1 + self.window_params['length'], 
                 xmax=t2 + self.window_params['length'],
                 linestyle='--', linewidth=2, color='royalblue', label='100°')
        ax.legend(loc=6, ncol=2, prop={'size': 10})
        ax.grid(linestyle='--', linewidth=0.5)
    
    def _add_colorbar(self, fig):
        """Add colorbar to the plot"""
        ax = plt.subplot2grid((3, 50), (2, 49))
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        mpl.colorbar.ColorbarBase(ax, cmap=plt.cm.RdYlGn_r,
                                norm=norm, orientation='vertical',
                                label="CC coefficient")
